#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import random
import configargparse
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from train import finetune, evaluate, pretrain, supervised
from datasets import get_dataloaders
from utils import *
import model.network as models
from model.moco import MoCo_Model


warnings.filterwarnings("ignore")


default_config = os.path.join(os.path.split(os.getcwd())[0], 'config.conf')

parser = configargparse.ArgumentParser(
    description='Pytorch MocoV2', default_config_files=[default_config])
parser.add_argument('-c', '--my-config', required=False,
                    is_config_file=True, help='config file path')
parser.add_argument('--dataset', default='cifar10',
                    help='Dataset, (Options: cifar10, cifar100, stl10, imagenet, tinyimagenet).')
parser.add_argument('--dataset_path', default=None,
                    help='Path to dataset, Not needed for TorchVision Datasets.')
parser.add_argument('--model', default='resnet18',
                    help='Model, (Options: resnet18, resnet34, resnet50, resnet101, resnet152).')
parser.add_argument('--n_epochs', type=int, default=1000,
                    help='Number of Epochs in Contrastive Training.')
parser.add_argument('--finetune_epochs', type=int, default=100,
                    help='Number of Epochs in Linear Classification Training.')
parser.add_argument('--warmup_epochs', type=int, default=10,
                    help='Number of Warmup Epochs During Contrastive Training.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of Samples Per Batch.')
parser.add_argument('--learning_rate', type=float, default=1.0,
                    help='Starting Learing Rate for Contrastive Training.')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='Base / Minimum Learing Rate to Begin Linear Warmup.')
parser.add_argument('--finetune_learning_rate', type=float, default=0.1,
                    help='Starting Learing Rate for Linear Classification Training.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='Contrastive Learning Weight Decay Regularisation Factor.')
parser.add_argument('--finetune_weight_decay', type=float, default=0.0,
                    help='Linear Classification Training Weight Decay Regularisation Factor.')
parser.add_argument('--optimiser', default='sgd',
                    help='Optimiser, (Options: sgd, adam, lars).')
parser.add_argument('--patience', default=50, type=int,
                    help='Number of Epochs to Wait for Improvement.')
parser.add_argument('--queue_size', type=int, default=65536,
                    help='Size of Memory Queue, Must be Divisible by batch_size.')
parser.add_argument('--queue_momentum', type=float, default=0.999,
                    help='Momentum for the Key Encoder Update.')
parser.add_argument('--temperature', type=float, default=0.07,
                    help='InfoNCE Temperature Factor')
parser.add_argument('--jitter_d', type=float, default=1.0,
                    help='Distortion Factor for the Random Colour Jitter Augmentation')
parser.add_argument('--jitter_p', type=float, default=0.8,
                    help='Probability to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_sigma', nargs=2, type=float, default=[0.1, 2.0],
                    help='Radius to Apply Random Colour Jitter Augmentation')
parser.add_argument('--blur_p', type=float, default=0.5,
                    help='Probability to Apply Gaussian Blur Augmentation')
parser.add_argument('--grey_p', type=float, default=0.2,
                    help='Probability to Apply Random Grey Scale')
parser.add_argument('--no_twocrop', dest='twocrop', action='store_false',
                    help='Whether or Not to Use Two Crop Augmentation, Used to Create Two Views of the Input for Contrastive Learning. (Default: True)')
parser.set_defaults(twocrop=True)
parser.add_argument('--load_checkpoint_dir', default=None,
                    help='Path to Load Pre-trained Model From.')
parser.add_argument('--no_distributed', dest='distributed', action='store_false',
                    help='Whether or Not to Use Distributed Training. (Default: True)')
parser.set_defaults(distributed=True)
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Perform Only Linear Classification Training. (Default: False)')
parser.set_defaults(finetune=False)
parser.add_argument('--supervised', dest='supervised', action='store_true',
                    help='Perform Supervised Pre-Training. (Default: False)')
parser.set_defaults(supervised=False)


def setup(distributed):
    """ Sets up for optional distributed training.
    For distributed training run as:
        python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=2 --use_env main.py
    To kill zombie processes use:
        kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
    For data parallel training on GPUs or CPU training run as:
        python main.py --no_distributed

    Taken from https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate

    args:
        distributed (bool):  Flag whether or not to perform distributed training.

    returns:
        local_rank (int): rank of local machine / host to perform distributed training.

        device (string): Device and rank of device to perform training on.

    """
    if distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ.get('LOCAL_RANK'))
        device = torch.device(f'cuda:{local_rank}')  # unique on individual node

        print('World size: {} ; Rank: {} ; LocalRank: {} ; Master: {}:{}'.format(
            os.environ.get('WORLD_SIZE'),
            os.environ.get('RANK'),
            os.environ.get('LOCAL_RANK'),
            os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')))
    else:
        local_rank = None
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = 44
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True

    return device, local_rank


def main():
    """ Main """

    # Arguments
    args = parser.parse_args()

    # Setup Distributed Training
    device, local_rank = setup(distributed=args.distributed)

    # Get Dataloaders for Dataset of choice
    dataloaders, args = get_dataloaders(args)

    # Setup logging, saving models, summaries
    args = experiment_config(parser, args)

    ''' Base Encoder '''

    # Get available models from /model/network.py
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    # If model exists
    if any(args.model in model_name for model_name in model_names):
        # Load model
        base_encoder = getattr(models, args.model)(
            args, num_classes=args.n_classes)  # Encoder

    else:
        raise NotImplementedError("Model Not Implemented: {}".format(args.model))

    if not args.supervised:
        # freeze all layers but the last fc
        for name, param in base_encoder.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        # init the fc layer
        init_weights(base_encoder)

    ''' MoCo Model '''
    moco = MoCo_Model(args, queue_size=args.queue_size,
                      momentum=args.queue_momentum, temperature=args.temperature)

    # Place model onto GPU(s)
    if args.distributed:
        torch.cuda.set_device(device)
        torch.set_num_threads(6)  # n cpu threads / n processes per node

        moco = DistributedDataParallel(moco.cuda(),
                                       device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
        base_encoder = DistributedDataParallel(base_encoder.cuda(),
                                               device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)

        # Only print from process (rank) 0
        args.print_progress = True if int(os.environ.get('RANK')) == 0 else False
    else:
        # If non Distributed use DataParallel
        if torch.cuda.device_count() > 1:
            moco = nn.DataParallel(moco)
            base_encoder = nn.DataParallel(base_encoder)

        print('\nUsing', torch.cuda.device_count(), 'GPU(s).\n')

        moco.to(device)
        base_encoder.to(device)

        args.print_progress = True

    # Print Network Structure and Params
    if args.print_progress:
        print_network(moco, args)  # prints out the network architecture etc
        logging.info('\npretrain/train: {} - valid: {} - test: {}'.format(
            len(dataloaders['train'].dataset), len(dataloaders['valid'].dataset),
            len(dataloaders['test'].dataset)))

    # launch model training or inference
    if not args.finetune:

        ''' Pretraining / Finetuning / Evaluate '''

        if not args.supervised:
            # Pretrain the encoder and projection head
            pretrain(moco, dataloaders, args)

            # Load the state_dict from query encoder and load it on finetune net
            base_encoder = load_moco(base_encoder, args)

        else:
            supervised(base_encoder, dataloaders, args)

            # Load the state_dict from query encoder and load it on finetune net
            base_encoder = load_sup(base_encoder, args)

        # Supervised Finetuning of the supervised classification head
        finetune(base_encoder, dataloaders, args)

        # Evaluate the pretrained model and trained supervised head
        test_loss, test_acc, test_acc_top5 = evaluate(
            base_encoder, dataloaders, 'test', args.finetune_epochs, args)

        print('[Test] loss {:.4f} - acc {:.4f} - acc_top5 {:.4f}'.format(
            test_loss, test_acc, test_acc_top5))

        if args.distributed:  # cleanup
            torch.distributed.destroy_process_group()
    else:

        ''' Finetuning / Evaluate '''

        # Do not Pretrain, just finetune and inference
        # Load the state_dict from query encoder and load it on finetune net
        base_encoder = load_moco(base_encoder, args)

        # Supervised Finetuning of the supervised classification head
        finetune(base_encoder, dataloaders, args)

        # Evaluate the pretrained model and trained supervised head
        test_loss, test_acc, test_acc_top5 = evaluate(
            base_encoder, dataloaders, 'test', args.finetune_epochs, args)

        print('[Test] loss {:.4f} - acc {:.4f} - acc_top5 {:.4f}'.format(
            test_loss, test_acc, test_acc_top5))

        if args.distributed:  # cleanup
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
