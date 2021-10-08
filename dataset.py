import torch
import torch.utils.data as data
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as torchdata
from myargs import args

class DsetThreeChannels(data.Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index):
        image, label = self.dset[index]
        return image.repeat(3, 1, 1), label

    def __len__(self):
        return len(self.dset)

def prepare_dataset(dataset_name, image_size, channels, path):

    #################################################################

    if dataset_name == 'usps':
        from usps import USPS
        tr_dataset = USPS(path+'/usps', 
            download=True, 
            train=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        te_dataset = USPS(path+'/usps', 
            download=True, 
            train=False, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        if channels == 3:
            tr_dataset = DsetThreeChannels(tr_dataset)
            te_dataset = DsetThreeChannels(te_dataset)

    #################################################################

    elif dataset_name == 'mnist':
        tr_dataset = torchvision.datasets.MNIST(path+'/mnist', 
            download=True, 
            train=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        te_dataset = torchvision.datasets.MNIST(path+'/mnist', 
            download=True, 
            train=False, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        if channels == 3:
            tr_dataset = DsetThreeChannels(tr_dataset)
            te_dataset = DsetThreeChannels(te_dataset)


    # elif dataset_name == 'mnist':
    #     tr_dataset = torchvision.datasets.MNIST(path+'/mnist', 
    #         download=True, 
    #         train=True, 
    #         transform=transforms.Compose([transforms.Resize(image_size), transforms.Grayscale(3)]))

    #     te_dataset = torchvision.datasets.MNIST(path+'/mnist', 
    #         download=True, 
    #         train=False, 
    #         transform=transforms.Compose([transforms.Resize(image_size), transforms.Grayscale(3)]))

    #     # if channels == 3:
    #     #     tr_dataset = DsetThreeChannels(tr_dataset)
    #     #     te_dataset = DsetThreeChannels(te_dataset)


    #################################################################

    # elif dataset_name == 'mnistrot':
    #     tr_dataset = torchvision.datasets.MNIST(path+'/mnistrot', 
    #         download=True, 
    #         train=True, 
    #         transform=transforms.Compose([
    #             transforms.Resize(image_size),
    #             transforms.RandomRotation(30, fill=(0,))]))

    #     te_dataset = torchvision.datasets.MNIST(path+'/mnistrot', 
    #         download=True, 
    #         train=False, 
    #         transform=transforms.Compose([transforms.Resize(image_size),
    #             transforms.RandomRotation(30, fill=(0,))]))

    #     if channels == 3:
    #         tr_dataset = DsetThreeChannels(tr_dataset)
    #         te_dataset = DsetThreeChannels(te_dataset)

    #################################################################

    elif dataset_name == 'mnistm':
        from mnistm import MNISTM
        tr_dataset = MNISTM(path+'/mnistm', 
            download=True, 
            train=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        te_dataset = MNISTM(path+'/mnistm', 
            download=True, 
            train=False, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

    #################################################################

    elif dataset_name == 'svhn':
        tr_dataset = torchvision.datasets.SVHN(root=path+'/svhn', 
            split='train', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        te_dataset = torchvision.datasets.SVHN(root=path+'/svhn', 
            split='test', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

    #################################################################

    elif dataset_name == 'svhn_extra':
        tr_dataset_basic = torchvision.datasets.SVHN(root=path+'/svhn', 
            split='train', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        print('SVHN basic set size: %d' %(len(tr_dataset_basic)))

        tr_dataset_extra = torchvision.datasets.SVHN(root=path+'/svhn', 
            split='extra', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        print('SVHN extra set size: %d' %(len(tr_dataset_extra)))

        tr_dataset = torchdata.ConcatDataset((tr_dataset_basic, tr_dataset_extra))

        te_dataset = torchvision.datasets.SVHN(root=path+'/svhn', 
            split='test', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

    #################################################################

    elif dataset_name == 'cifar10':
        tr_dataset = torchvision.datasets.CIFAR10(root = path+'/cifar10', 
            train=True, 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        te_dataset = torchvision.datasets.CIFAR10(root = path+'/cifar10', 
            train=False, 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        from modify_cifar_stl import modify_cifar
        modify_cifar(tr_dataset)
        modify_cifar(te_dataset)

    #################################################################

    elif dataset_name == 'stl':
        tr_dataset = torchvision.datasets.STL10(root = path+'/', 
            split='train', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        te_dataset = torchvision.datasets.STL10(root = path+'/', 
            split='test', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        from modify_cifar_stl import modify_stl
        modify_stl(tr_dataset)
        modify_stl(te_dataset)

    #################################################################

    else:
        raise ValueError('Dataset %s not found!' %(dataset_name))

    #################################################################
    
    return tr_dataset, te_dataset

class create_dataset(data.Dataset):
    def __init__(self, args):

        sc_tr_dataset, sc_te_dataset = prepare_dataset(args.source, args.image_size, args.channels, path=args.dset_dir)
        tg_tr_dataset, tg_te_dataset = prepare_dataset(args.target, args.image_size, args.channels, path=args.dset_dir)
        
        self.datalist_src = sc_tr_dataset
        self.datalist_target = tg_tr_dataset


        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.source_larger = len(self.datalist_src) > len(self.datalist_target)
        self.n_smallerdataset = len(self.datalist_target) if self.source_larger else len(self.datalist_src)


    def __len__(self):
        return np.maximum(len(self.datalist_src), len(self.datalist_target))

    def shuffledata(self):
        self.datalist_src = [self.datalist_src[ij] for ij in torch.randperm(len(self.datalist_src))]
        self.datalist_target = [self.datalist_target[ij] for ij in torch.randperm(len(self.datalist_target))]

    def __getitem__(self, index):
        index_src = index if self.source_larger else index % self.n_smallerdataset
        index_target = index if not self.source_larger else index % self.n_smallerdataset

        image_source, label_source = self.datalist_src[index_src]
        image_source = self.totensor(image_source)
        image_source = self.normalize(image_source)

        image_target, label_target = self.datalist_target[index_target]
        image_target = self.totensor(image_target)
        image_target = self.normalize(image_target)        

        return image_source, label_source, image_target, label_target

class create_dataset_eval(data.Dataset):
    def __init__(self, args):


        tg_tr_dataset, tg_te_dataset = prepare_dataset(args.target, args.image_size, args.channels, path=args.dset_dir)
        

        self.datalist_target = tg_te_dataset

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.datalist_target)


    def __getitem__(self, index):

        image_target, label_target = self.datalist_target[index]
        image_target = self.totensor(image_target)
        image_target = self.normalize(image_target)        

        return image_target, label_target

def GenerateIterator_eval(args):
    params = {
        'pin_memory': True,
        'batch_size': args.batch_size_eval,
        'num_workers': args.workers,
    }
    return data.DataLoader(create_dataset_eval(args), **params)

def GenerateIterator(args):
    params = {
        'pin_memory': True,
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'drop_last': True,
    }
    return data.DataLoader(create_dataset(args), **params)
