# -*- coding: utf-8 -*-
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, STL10, ImageNet, CIFAR100, ImageFolder

from utils import *


def get_dataloaders(args):
    '''
    Retrives the dataloaders for the dataset of choice.

    Initalise variables that correspond to the dataset of choice.

    args:
        args (dict): Program arguments/commandline arguments.

    returns:
        dataloaders (dict): pretrain,train,valid,train_valid,test set split dataloaders.

        args (dict): Updated and Additional program/commandline arguments dependent on dataset.

    '''
    if args.dataset == 'cifar10':
        dataset = 'CIFAR10'

        args.class_names = (
            'plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
        )  # 0,1,2,3,4,5,6,7,8,9 labels

        args.crop_dim = 32
        args.n_channels, args.n_classes = 3, 10

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test')}

        dataloaders = cifar_dataloader(args, dataset_paths)

    elif args.dataset == 'cifar100':
        dataset = 'CIFAR100'

        args.class_names = None

        args.crop_dim = 32
        args.n_channels, args.n_classes = 3, 100

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test')}

        dataloaders = cifar_dataloader(args, dataset_paths)

    elif args.dataset == 'stl10':
        dataset = 'STL10'

        args.class_names = None

        args.crop_dim = 96
        args.n_channels, args.n_classes = 3, 10

        # Get and make dir to download dataset to.
        working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        dataset_paths = {'train': os.path.join(working_dir, 'train'),
                         'test':  os.path.join(working_dir, 'test'),
                         'pretrain':  os.path.join(working_dir, 'unlabeled')}

        dataloaders = stl10_dataloader(args, dataset_paths)

    elif args.dataset == 'imagenet':
        dataset = 'ImageNet'

        args.class_names = None

        args.crop_dim = 224
        args.n_channels, args.n_classes = 3, 1000

        # Get and make dir to download dataset to.
        target_dir = args.dataset_path

        if not target_dir is None:
            dataset_paths = {'train': os.path.join(target_dir, 'train'),
                             'test':  os.path.join(target_dir, 'val')}

            dataloaders = imagenet_dataloader(args, dataset_paths)

        else:
            NotImplementedError('Please Select a path for the {} Dataset.'.format(args.dataset))

    elif args.dataset == 'tinyimagenet':
        dataset = 'TinyImageNet'

        args.class_names = None

        args.crop_dim = 64
        args.n_channels, args.n_classes = 3, 200

        # Get and make dir to download dataset to.
        target_dir = args.dataset_path

        if not target_dir is None:
            dataset_paths = {'train': os.path.join(target_dir, 'train'),
                             'test':  os.path.join(target_dir, 'val')}

            dataloaders = imagenet_dataloader(args, dataset_paths)

        else:
            NotImplementedError('Please Select a path for the {} Dataset.'.format(args.dataset))
    else:
        NotImplementedError('{} dataset not available.'.format(args.dataset))

    return dataloaders, args


def imagenet_dataloader(args, dataset_paths):
    '''
    Loads the ImageNet or TinyImageNet dataset performing augmentaions.

    Generates splits of the training set to produce a validation set.

    args:
        args (dict): Program/commandline arguments.

        dataset_paths (dict): Paths to each datset split.

    Returns:

        dataloaders (): pretrain,train,valid,train_valid,test set split dataloaders.
    '''

    # guassian_blur from https://github.com/facebookresearch/moco/
    guassian_blur = transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p)

    color_jitter = transforms.ColorJitter(
        0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=args.jitter_p)

    rnd_grey = transforms.RandomGrayscale(p=args.grey_p)

    # Base train and test augmentaions
    transf = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim)),
            rnd_color_jitter,
            rnd_grey,
            guassian_blur,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))]),
        'test':  transforms.Compose([
            transforms.CenterCrop((args.crop_dim, args.crop_dim)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    }

    config = {'train': True, 'test': False}

    datasets = {i: ImageFolder(root=dataset_paths[i]) for i in config.keys()}

    # weighted sampler weights for full(f) training set
    f_s_weights = sample_weights(datasets['train'].targets)

    # return data, labels dicts for new train set and class-balanced valid set
    # 50 is the num of samples to be split into the val set for each class (1000)
    data, labels = random_split_image_folder(data=np.asarray(datasets['train'].samples),
                                             labels=datasets['train'].targets,
                                             n_classes=args.n_classes,
                                             n_samples_per_class=np.repeat(50, args.n_classes).reshape(-1))

    # torch.from_numpy(np.stack(labels)) this takes the list of class ids and turns them to tensor.long

    # original full training set
    datasets['train_valid'] = CustomDataset(data=np.asarray(datasets['train'].samples),
                                            labels=torch.from_numpy(np.stack(datasets['train'].targets)), transform=transf['train'], two_crop=args.twocrop)

    # original test set
    datasets['test'] = CustomDataset(data=np.asarray(datasets['test'].samples),
                                     labels=torch.from_numpy(np.stack(datasets['test'].targets)), transform=transf['test'], two_crop=False)

    # make new pretraining set without validation samples
    datasets['pretrain'] = CustomDataset(data=np.asarray(data['train']),
                                         labels=labels['train'], transform=transf['train'], two_crop=args.twocrop)

    # make new finetuning set without validation samples
    datasets['train'] = CustomDataset(data=np.asarray(data['train']),
                                      labels=labels['train'], transform=transf['train'], two_crop=False)

    # make class balanced validation set for finetuning
    datasets['valid'] = CustomDataset(data=np.asarray(data['valid']),
                                      labels=labels['valid'], transform=transf['test'], two_crop=False)

    # weighted sampler weights for new training set
    s_weights = sample_weights(datasets['pretrain'].labels)

    config = {
        'pretrain': WeightedRandomSampler(s_weights,
                                          num_samples=len(s_weights), replacement=True),
        'train': WeightedRandomSampler(s_weights,
                                       num_samples=len(s_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_s_weights,
                                             num_samples=len(f_s_weights), replacement=True),
        'valid': None, 'test': None
    }

    if args.distributed:
        config = {'pretrain': DistributedSampler(datasets['pretrain']),
                  'train': DistributedSampler(datasets['train']),
                  'train_valid': DistributedSampler(datasets['train_valid']),
                  'valid': None, 'test': None}

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
                                 num_workers=8, pin_memory=True, drop_last=True,
                                 batch_size=args.batch_size) for i in config.keys()}

    return dataloaders


def stl10_dataloader(args, dataset_paths):
    '''
    Loads the STL10 dataset performing augmentaions.

    Generates splits of the training set to produce a validation set.

    args:
        args (dict): Program/commandline arguments.

        dataset_paths (dict): Paths to each datset split.

    Returns:

        dataloaders (): pretrain,train,valid,train_valid,test set split dataloaders.
    '''

    # guassian_blur from https://github.com/facebookresearch/moco/
    guassian_blur = transforms.RandomApply([GaussianBlur(args.blur_sigma)], p=args.blur_p)

    color_jitter = transforms.ColorJitter(
        0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=args.jitter_p)

    rnd_grey = transforms.RandomGrayscale(p=args.grey_p)

    # Base train and test augmentaions
    transf = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            rnd_grey,
            guassian_blur,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))]),
        'valid':  transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))]),
        'test':  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))])
    }

    transf['pretrain'] = transf['train']

    config = {'train': 'train', 'test': 'test', 'pretrain': 'unlabeled'}

    datasets = {i: STL10(root=dataset_paths[i], transform=transf[i],
                         split=config[i], download=True) for i in config.keys()}

    # weighted sampler weights for full(f) training set
    f_s_weights = sample_weights(datasets['train'].labels)

    # return data, labels dicts for new train set and class-balanced valid set
    # 500 is the num of samples to be split into the val set for each class (10)
    data, labels = random_split(data=datasets['train'].data,
                                labels=datasets['train'].labels,
                                n_classes=args.n_classes,
                                n_samples_per_class=np.repeat(50, args.n_classes).reshape(-1))

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new pretraining set without validation samples
    datasets['pretrain'] = CustomDataset(data=datasets['pretrain'].data,
                                         labels=None, transform=transf['pretrain'], two_crop=args.twocrop)

    # make new finetuning set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
                                      labels=labels['train'], transform=transf['train'], two_crop=False)

    # make class balanced validation set for finetuning
    datasets['valid'] = CustomDataset(data=data['valid'],
                                      labels=labels['valid'], transform=transf['valid'], two_crop=False)

    # weighted sampler weights for new training set
    s_weights = sample_weights(datasets['train'].labels)

    config = {
        'pretrain': None,
        'train': WeightedRandomSampler(s_weights,
                                       num_samples=len(s_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_s_weights,
                                             num_samples=len(f_s_weights), replacement=True),
        'valid': None, 'test': None
    }

    if args.distributed:
        config = {'pretrain': DistributedSampler(datasets['pretrain']),
                  'train': DistributedSampler(datasets['train']),
                  'train_valid': DistributedSampler(datasets['train_valid']),
                  'valid': None, 'test': None}

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
                                 num_workers=8, pin_memory=True, drop_last=True,
                                 batch_size=args.batch_size) for i in config.keys()}

    return dataloaders


def cifar_dataloader(args, dataset_paths):
    '''
    Loads the CIFAR10 or CIFAR100 dataset performing augmentaions.

    Generates splits of the training set to produce a validation set.

    args:
        args (dict): Program/commandline arguments.

        dataset_paths (dict): Paths to each datset split.

    Returns:

        dataloaders (): pretrain,train,valid,train_valid,test set split dataloaders.
    '''

    color_jitter = transforms.ColorJitter(
        0.8*args.jitter_d, 0.8*args.jitter_d, 0.8*args.jitter_d, 0.2*args.jitter_d)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=args.jitter_p)

    rnd_grey = transforms.RandomGrayscale(p=args.grey_p)

    # Base train and test augmentaions
    transf = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            rnd_grey,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim), scale=(0.25, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))]),
        'pretrain': transforms.Compose([
            transforms.ToPILImage(),
            rnd_color_jitter,
            rnd_grey,
            transforms.RandomResizedCrop((args.crop_dim, args.crop_dim)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))]),
        'test':  transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215841, 0.44653091),
                                 (0.24703223, 0.24348513, 0.26158784))])
    }

    config = {'train': True, 'test': False}

    if args.dataset == 'cifar10':

        datasets = {i: CIFAR10(root=dataset_paths[i], transform=transf[i],
                               train=config[i], download=True) for i in config.keys()}
        val_samples = 500

    elif args.dataset == 'cifar100':

        datasets = {i: CIFAR100(root=dataset_paths[i], transform=transf[i],
                                train=config[i], download=True) for i in config.keys()}

        val_samples = 100

    # weighted sampler weights for full(f) training set
    f_s_weights = sample_weights(datasets['train'].targets)

    # return data, labels dicts for new train set and class-balanced valid set
    # 500 is the num of samples to be split into the val set for each class (10)
    data, labels = random_split(data=datasets['train'].data,
                                labels=datasets['train'].targets,
                                n_classes=args.n_classes,
                                n_samples_per_class=np.repeat(val_samples, args.n_classes).reshape(-1))

    # save original full training set
    datasets['train_valid'] = datasets['train']

    # make new pretraining set without validation samples
    datasets['pretrain'] = CustomDataset(data=data['train'],
                                         labels=labels['train'], transform=transf['pretrain'], two_crop=args.twocrop)

    # make new finetuning set without validation samples
    datasets['train'] = CustomDataset(data=data['train'],
                                      labels=labels['train'], transform=transf['train'], two_crop=False)

    # make class balanced validation set for finetuning
    datasets['valid'] = CustomDataset(data=data['valid'],
                                      labels=labels['valid'], transform=transf['test'], two_crop=False)

    # weighted sampler weights for new training set
    s_weights = sample_weights(datasets['pretrain'].labels)

    config = {
        'pretrain': WeightedRandomSampler(s_weights,
                                          num_samples=len(s_weights), replacement=True),
        'train': WeightedRandomSampler(s_weights,
                                       num_samples=len(s_weights), replacement=True),
        'train_valid': WeightedRandomSampler(f_s_weights,
                                             num_samples=len(f_s_weights), replacement=True),
        'valid': None, 'test': None
    }

    if args.distributed:
        config = {'pretrain': DistributedSampler(datasets['pretrain']),
                  'train': DistributedSampler(datasets['train']),
                  'train_valid': DistributedSampler(datasets['train_valid']),
                  'valid': None, 'test': None}

    dataloaders = {i: DataLoader(datasets[i], sampler=config[i],
                                 num_workers=8, pin_memory=True, drop_last=True,
                                 batch_size=args.batch_size) for i in config.keys()}

    return dataloaders
