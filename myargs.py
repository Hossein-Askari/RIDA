import argparse

parser = argparse.ArgumentParser()




######################## Model parameters ########################

parser.add_argument('--classes', default=9, type=int, help='# of classes')
parser.add_argument("--decay", default=100, dest='decay', type=float, help="number of epochs to decay the lr by half")

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.0000, type=float, help='weight decay/weights regularizer for sgd')
parser.add_argument('--beta1', default=0.5, type=float, help='momentum for sgd, beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float, help=', beta2 for adam')
parser.add_argument('--large', default=True, type=bool, help=', use large network')

parser.add_argument('--num_epoch', default=2000, type=int, help='epochs to train for')
parser.add_argument('--start_epoch', default=1, type=int, help='epoch to start training. useful if continue from a checkpoint')

parser.add_argument('--batch_size', default=64, type=int, help='input batch size')
parser.add_argument('--batch_size_eval', default=64, type=int, help='input batch size at eval time')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--gpu_ids', default='0', help='which gpus to use in train/eval')

parser.add_argument('--radius', type=float, default=3.5, help="Perturbation 2-norm ball radius")
parser.add_argument('--gaussian_noise', type=float, default=1.0, help="noise for feature extractor")
parser.add_argument('--n_power', type=int, default=1, help="gradient iterations")



parser.add_argument('--num_latent',  default=192*8*8, type=int, help="dimension of latent code z")
parser.add_argument('--feature_size',  default=192, type=int, help="dimension of latent code z")
parser.add_argument('--fc_dim',  default=512, type=int, help="dimension of FC layer in the flow")
parser.add_argument('--num_block',  default=4, type=int, help="number of affine coupling layers in the flow")


######################## Model paths ########################

parser.add_argument('--source', default='stl', type = str, help='# of choises MNIST,SVHN')
parser.add_argument('--target', default='cifar10', type = str, help='# of choises MNIST,SVHN')
parser.add_argument('--dataset_name', default='mnisttosvhn', type = str, help='# of choises MNIST,SVHN')
parser.add_argument('--channels',  default=3, type=int, help="size of training image")
parser.add_argument('--image_size',  default=32, type=int, help="size of training image")
parser.add_argument('--dset_dir', dest='dset_dir', default="../data", help='Where you store the dataset.')


args = parser.parse_args()
