import torch
from torch import nn
from dataset import GenerateIterator, GenerateIterator_eval
from myargs import args
import numpy as np
from tqdm import tqdm
from models import Encoder, Decoder, Classifier, Discriminator, EMA
from vat import VAT, ConditionalEntropyLoss

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom


from torch.utils.tensorboard import SummaryWriter

###############################################
# experiments
###############################################

experiment = '1'
writer_tensorboard = SummaryWriter('tb2_loggers/' + experiment) 


# discriminator network
feature_discriminator = Discriminator(large=args.large).cuda()

# classifier network.
encoder = Encoder(args).cuda()
classifier = Classifier(args).cuda()

##################################################################

# batchnorm_args = {'track_running_stats':True,}

def subnet_fc(c_in, c_out):
	return nn.Sequential(nn.Linear(c_in, args.fc_dim), nn.ReLU(), nn.BatchNorm1d(args.fc_dim),
		nn.Linear(args.fc_dim,  c_out))    

nodes = [InputNode(args.num_latent, name='input')]
for k in range(args.num_block):
	nodes.append(Node(nodes[-1],
					  GLOWCouplingBlock,
					  {'subnet_constructor':subnet_fc, 'clamp':2.0},
					  name=F'coupling_{k}'))
	nodes.append(Node(nodes[-1],
					  PermuteRandom,
					  {'seed':k},
					  name=F'permute_{k}'))


nodes.append(OutputNode(nodes[-1], name='output'))
Flow = ReversibleGraphNet(nodes, verbose=False)

modFlow = Flow.cuda()

##################################################################

# def subnet_conv(c_in, c_out):
#     return nn.Sequential(nn.Conv2d(c_in, 256,   192, padding=1), nn.ReLU(),
#                          nn.Conv2d(256,  c_out, 192, padding=1))


# nodes = [InputNode(192, 8, 8, name='input')]
# ndim_x = 192 * 8 * 8

# for k in range(args.num_block):
#    nodes.append(Node(nodes[-1],
#                         GLOWCouplingBlock,
#                         {'subnet_constructor':subnet_conv, 'clamp':1.2},
#                         name=F'conv_high_res_{k}'))
#    nodes.append(Node(nodes[-1],
#                         PermuteRandom,
#                         {'seed':k},
#                         name=F'permute_high_res_{k}'))

# nodes.append(OutputNode(nodes[-1], name='output'))
# Flow = ReversibleGraphNet(nodes, verbose=False)
# modFlow = Flow.cuda()


###############################################

# loss functions
cent = ConditionalEntropyLoss().cuda()
xent = nn.CrossEntropyLoss(reduction='mean').cuda()
sigmoid_xent = nn.BCEWithLogitsLoss(reduction='mean').cuda()

###############################################

optimizer_ecd = torch.optim.Adam([{'params': encoder.parameters()}, 
								  {'params': classifier.parameters()}], lr=args.lr, betas=(args.beta1, args.beta2))


optimizer_disc = torch.optim.Adam(feature_discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))


trainable_parameters = [p for p in modFlow.parameters() if p.requires_grad]
optimizer_flw = torch.optim.Adam(trainable_parameters, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-6, weight_decay=2e-5)
for param in trainable_parameters:
	param.data = 0.05*torch.randn_like(param)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer_ecd, args.decay, gamma=0.5, last_epoch=-1)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_disc, args.decay, gamma=0.5, last_epoch=-1)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer_flw, args.decay, gamma=0.5, last_epoch=-1)


###############################################

# datasets.
iterator_train = GenerateIterator(args)
iterator_val = GenerateIterator_eval(args)

# loss params.

dw = 0
cw = 0
sw = 1
tw = 1e-1
bw = 1e-2

ema = EMA(0.998)
ema.register(encoder)
ema.register(classifier)

# training..
for epoch in range(1, args.num_epoch):
	iterator_train.dataset.shuffledata()
	pbar = tqdm(iterator_train, disable=False,
				bar_format="{percentage:.0f}%,{elapsed},{remaining},{desc}")

	loss_main_sum, n_total = 0, 0
	loss_domain_sum, loss_src_class_sum, \
	loss_src_vat_sum, loss_trg_cent_sum, loss_trg_vat_sum = 0, 0, 0, 0, 0
	loss_disc_sum = 0
	loss_src_rec_sum = 0
	loss_trg_rec_sum = 0
	
	l_nll_s_sum = 0
	l_nll_t_sum = 0
	log_d_s_sum = 0 
	log_d_t_sum = 0

	for images_source, labels_source, images_target, labels_target in pbar:
		images_source, labels_source, images_target, labels_target = images_source.cuda(), labels_source.cuda(), images_target.cuda(), labels_target.cuda()

		# pass images through the classifier network.
		feats_source = encoder(images_source)
		feats_src_hat, log_d_s = modFlow(feats_source) 
		
		# flow loss 
		loss_ll_s = log_d_s
		l_nll_s = -loss_ll_s.mean()

		pred_source = classifier(feats_src_hat)

		feats_target = encoder(images_target, track_bn=True)

		feats_trg_hat, log_d_t = modFlow(feats_target, ) 
		
		# flow loss 
		loss_ll_t = log_d_t
		l_nll_t = -loss_ll_t.mean()

		pred_target = classifier(feats_trg_hat, track_bn=True)


		' Discriminator losses setup. '
		# discriminator loss.
		real_logit_disc = feature_discriminator(feats_source.detach())
		fake_logit_disc = feature_discriminator(feats_target.detach())

		loss_disc = 0.5 * (
				sigmoid_xent(real_logit_disc, torch.ones_like(real_logit_disc, device='cuda')) +
				sigmoid_xent(fake_logit_disc, torch.zeros_like(fake_logit_disc, device='cuda'))
		)

		' Classifier losses setup. '
		# supervised/source classification.
		loss_src_class = xent(pred_source, labels_source)

		# conditional entropy loss.
		loss_trg_cent = cent(pred_target)

		# domain loss.
		real_logit = feature_discriminator(feats_source)
		fake_logit = feature_discriminator(feats_target)

		loss_domain = 0.5 * (
				sigmoid_xent(real_logit, torch.zeros_like(real_logit, device='cuda')) +
				sigmoid_xent(fake_logit, torch.ones_like(fake_logit, device='cuda'))
		)

		# combined loss.
		loss_main = (
				dw * loss_domain +
				cw * loss_src_class +
				tw * loss_trg_cent +
				0 * l_nll_s +
				1 * l_nll_t
		)

		' Update network(s) '

		# Update discriminator.
		optimizer_disc.zero_grad()
		loss_disc.backward()
		optimizer_disc.step()

		optimizer_flw.zero_grad()
		optimizer_ecd.zero_grad()
		loss_main.backward()
		# torch.nn.utils.clip_grad_norm_(trainable_parameters, 1, norm_type=2)
		optimizer_ecd.step()
		optimizer_flw.step()

		# Polyak averaging.
		ema(encoder)  # TODO: move ema into the optimizer step fn.
		ema(classifier) 

		l_nll_s_sum += l_nll_s
		l_nll_t_sum += l_nll_t
		log_d_s_sum += log_d_s
		log_d_t_sum += log_d_t



		loss_domain_sum += loss_domain.item()
		loss_src_class_sum += loss_src_class.item()
		loss_trg_cent_sum += loss_trg_cent.item()
		loss_main_sum += loss_main.item()
		loss_disc_sum += loss_disc.item()
		n_total += 1

		pbar.set_description('loss {:.3f},'
							 ' domain {:.3f},'
							 ' s cls {:.3f},'
							 ' t c-ent {:.3f},'
							 ' disc {:.3f}'.format(
			loss_main_sum/n_total,
			loss_domain_sum/n_total,
			loss_src_class_sum/n_total,
			loss_trg_cent_sum/n_total,
			loss_disc_sum / n_total,
		)
	)

	scheduler1.step()
	scheduler2.step() 
	scheduler3.step() 
	
	# validate.
	if epoch % 1 == 0:

		encoder.eval()
		classifier.eval()
		feature_discriminator.eval()
		modFlow.eval()
		
		with torch.no_grad():
			preds_val, gts_val = [], []
			val_loss = 0
			for images_target, labels_target in iterator_val:
				images_target, labels_target = images_target.cuda(), labels_target.cuda()

				# cross entropy based classification
				feats = encoder(images_target)
				sss,_ = modFlow(feats) 
				pred_val = classifier(sss)


				pred_val = np.argmax(pred_val.cpu().data.numpy(), 1)
				preds_val.extend(pred_val)
				gts_val.extend(labels_target)

			preds_val = np.asarray(preds_val)
			gts_val = np.asarray(gts_val)

			score_cls_val = (np.mean(preds_val == gts_val)).astype(np.float)
			print('\n({}) acc. v {:.3f}\n'.format(epoch, score_cls_val))
		
		encoder.train()
		classifier.train()
		feature_discriminator.train()
		modFlow.train()


	# writer_tensorboard.add_scalar('nll_s', l_nll_s_sum/n_total, epoch)
	# writer_tensorboard.add_scalar('logd_s', log_d_s_sum/n_total, epoch)
	# writer_tensorboard.add_scalar('nll_t', l_nll_t_sum/n_total, epoch)
	# writer_tensorboard.add_scalar('logd_t', log_d_t_sum/n_total, epoch)

	# writer_tensorboard.add_scalar('logd_t', np.mean(log_D_t), epoch)
	# writer_tensorboard.add_scalar('nll_t', np.mean(like_losses_t), epoch)
	# writer_tensorboard.add_scalar('s_to_t', np.mean(l1_loss), epoch)
	# writer_tensorboard.add_scalar('cls', np.mean(cls_losses_s), epoch)
	# writer_tensorboard.add_scalar('ce', np.mean(ce_losses_t), epoch)
	# writer_tensorboard.add_scalar('src_acc', np.mean(src_acc), epoch)



	
	writer_tensorboard.add_scalar('trg_val', score_cls_val, epoch)