# -*- coding: utf-8 -*-
import os
import gc
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from optimisers import get_optimiser
from PIL import Image


def pretrain(encoder, dataloaders, args):
    ''' Pretrain script - MoCo

        Pretrain the encoder and projection head with a Contrastive InfoNCE Loss.
    '''
    mode = 'pretrain'

    ''' Optimisers '''
    optimiser = get_optimiser((encoder,), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = args.base_lr

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.0001, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(optimiser, args.n_epochs)

    ''' Loss / Criterion '''
    criterion = nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    patience_counter = 0

    ''' Pretrain loop '''
    for epoch in range(args.n_epochs):

        # Train models
        encoder.train()

        sample_count = 0
        run_loss = 0

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['pretrain'])
        else:
            train_dataloader = dataloaders['pretrain']

        ''' epoch loop '''
        for i, (inputs, _) in enumerate(train_dataloader):

            inputs = inputs.cuda(non_blocking=True)

            # Forward pass
            optimiser.zero_grad()

            # retrieve the 2 views
            x_i, x_j = torch.split(inputs, [3, 3], dim=1)

            # Get the encoder representation
            logit, label = encoder(x_i, x_j)

            loss = criterion(logit, label)

            loss.backward()

            optimiser.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)

            run_loss += loss.item()

        epoch_pretrain_loss = run_loss / len(dataloaders['pretrain'])

        ''' Update Schedulers '''
        # TODO: Improve / add lr_scheduler for warmup
        if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
            wu_lr = (args.learning_rate - args.base_lr) * \
                (float(epoch+1) / args.warmup_epochs) + args.base_lr
            save_lr = optimiser.param_groups[0]['lr']
            optimiser.param_groups[0]['lr'] = wu_lr
        else:
            # After warmup, decay lr with CosineAnnealingLR
            lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Train] loss: {:.4f}'.format(epoch_pretrain_loss))

            args.writer.add_scalars('epoch_loss', {'pretrain': epoch_pretrain_loss}, epoch+1)
            args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if epoch_pretrain_loss <= best_valid_loss:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_loss = epoch_pretrain_loss

            # saving using process (rank) 0 only as all processes are in sync

            state = {
                #'args': args,
                'moco': encoder.state_dict(),
                'optimiser': optimiser.state_dict(),
                'epoch': epoch,
            }

            torch.save(state, args.checkpoint_dir)
        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                break

        epoch_pretrain_loss = None  # reset loss

    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory


def supervised(encoder, dataloaders, args):
    ''' Supervised Train script - MoCo

        Supervised Training encoder and train the supervised classification head with a Cross Entropy Loss.
    '''

    mode = 'pretrain'

    ''' Optimisers '''
    # Only optimise the supervised head
    optimiser = get_optimiser((encoder,), mode, args)

    ''' Schedulers '''
    # Warmup Scheduler
    if args.warmup_epochs > 0:
        for param_group in optimiser.param_groups:
            param_group['lr'] = args.base_lr

        # Cosine LR Decay after the warmup epochs
        lr_decay = lr_scheduler.CosineAnnealingLR(
            optimiser, (args.n_epochs-args.warmup_epochs), eta_min=0.0001, last_epoch=-1)
    else:
        # Cosine LR Decay
        lr_decay = lr_scheduler.CosineAnnealingLR(optimiser, args.n_epochs)

    ''' Loss / Criterion '''
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    patience_counter = 0

    ''' Pretrain loop '''
    for epoch in range(args.n_epochs):

        # Train models
        encoder.train()

        sample_count = 0
        run_loss = 0
        run_top1 = 0.0
        run_top5 = 0.0

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.n_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['train'])
        else:
            train_dataloader = dataloaders['train']

        ''' epoch loop '''
        for i, (inputs, target) in enumerate(train_dataloader):

            inputs = inputs.cuda(non_blocking=True)

            target = target.cuda(non_blocking=True)

            # Forward pass
            optimiser.zero_grad()

            output = encoder(inputs)

            loss = criterion(output, target)

            loss.backward()

            optimiser.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)

            run_loss += loss.item()

            predicted = output.argmax(1)

            acc = (predicted == target).sum().item() / target.size(0)

            run_top1 += acc

            _, output_topk = output.topk(5, 1, True, True)

            acc_top5 = (output_topk == target.view(-1, 1).expand_as(output_topk)
                        ).sum().item() / target.size(0)  # num corrects

            run_top5 += acc_top5

        epoch_pretrain_loss = run_loss / len(dataloaders['train'])  # sample_count

        epoch_pretrain_acc = run_top1 / len(dataloaders['train'])

        epoch_pretrain_acc_top5 = run_top5 / len(dataloaders['train'])

        ''' Update Schedulers '''
        # TODO: Improve / add lr_scheduler for warmup
        if args.warmup_epochs > 0 and epoch+1 <= args.warmup_epochs:
            wu_lr = (args.learning_rate - args.base_lr) * \
                (float(epoch+1) / args.warmup_epochs) + args.base_lr
            save_lr = optimiser.param_groups[0]['lr']
            optimiser.param_groups[0]['lr'] = wu_lr
        else:
            # After warmup, decay lr with CosineAnnealingLR
            lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Train] loss: {:.4f}'.format(epoch_pretrain_loss))

            args.writer.add_scalars('epoch_loss', {
                                    'pretrain': epoch_pretrain_loss}, epoch+1)
            args.writer.add_scalars('supervised_epoch_acc', {
                                    'pretrain': epoch_pretrain_acc}, epoch+1)
            args.writer.add_scalars('supervised_epoch_acc_top5', {
                                    'pretrain': epoch_pretrain_acc_top5}, epoch+1)
            args.writer.add_scalars('epoch_loss', {'pretrain': epoch_pretrain_loss}, epoch+1)
            args.writer.add_scalars('lr', {'pretrain': optimiser.param_groups[0]['lr']}, epoch+1)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if epoch_pretrain_loss <= best_valid_loss:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_loss = epoch_pretrain_loss

            # saving using process (rank) 0 only as all processes are in sync

            state = {
                #'args': args,
                'encoder': encoder.state_dict(),
                'optimiser': optimiser.state_dict(),
                'epoch': epoch,
            }

            torch.save(state, args.checkpoint_dir)
        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                break

        epoch_pretrain_loss = None  # reset loss

    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory


def finetune(encoder, dataloaders, args):
    ''' Finetune script - MoCo

        Freeze the encoder and train the supervised Linear Evaluation head with a Cross Entropy Loss.
    '''

    mode = 'finetune'

    ''' Optimisers '''
    # Only optimise the supervised head
    optimiser = get_optimiser((encoder,), mode, args)

    ''' Schedulers '''
    # Cosine LR Decay
    lr_decay = lr_scheduler.CosineAnnealingLR(optimiser, args.finetune_epochs)

    ''' Loss / Criterion '''
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)
    best_valid_loss = np.inf
    best_valid_acc = 0.0
    patience_counter = 0

    ''' Pretrain loop '''
    for epoch in range(args.finetune_epochs):

        # Freeze the encoder, train classification head
        encoder.eval()

        sample_count = 0
        run_loss = 0
        run_top1 = 0.0
        run_top5 = 0.0

        # Print setup for distributed only printing on one node.
        if args.print_progress:
            logging.info('\nEpoch {}/{}:\n'.format(epoch+1, args.finetune_epochs))
            # tqdm for process (rank) 0 only when using distributed training
            train_dataloader = tqdm(dataloaders['train'])
        else:
            train_dataloader = dataloaders['train']

        ''' epoch loop '''
        for i, (inputs, target) in enumerate(train_dataloader):

            inputs = inputs.cuda(non_blocking=True)

            target = target.cuda(non_blocking=True)

            # Forward pass
            optimiser.zero_grad()

            # Do not compute the gradients for the frozen encoder
            output = encoder(inputs)

            # Take pretrained encoder representations
            loss = criterion(output, target)

            loss.backward()

            optimiser.step()

            torch.cuda.synchronize()

            sample_count += inputs.size(0)

            run_loss += loss.item()

            predicted = output.argmax(1)

            acc = (predicted == target).sum().item() / target.size(0)

            run_top1 += acc

            _, output_topk = output.topk(5, 1, True, True)

            acc_top5 = (output_topk == target.view(-1, 1).expand_as(output_topk)
                        ).sum().item() / target.size(0)  # num corrects

            run_top5 += acc_top5

        epoch_finetune_loss = run_loss / len(dataloaders['train'])  # sample_count

        epoch_finetune_acc = run_top1 / len(dataloaders['train'])

        epoch_finetune_acc_top5 = run_top5 / len(dataloaders['train'])

        ''' Update Schedulers '''
        # Decay lr with CosineAnnealingLR
        lr_decay.step()

        ''' Printing '''
        if args.print_progress:  # only validate using process 0
            logging.info('\n[Finetune] loss: {:.4f},\t acc: {:.4f}, \t acc_top5: {:.4f}\n'.format(
                epoch_finetune_loss, epoch_finetune_acc, epoch_finetune_acc_top5))

            args.writer.add_scalars('finetune_epoch_loss', {'train': epoch_finetune_loss}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc', {'train': epoch_finetune_acc}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc_top5', {
                                    'train': epoch_finetune_acc_top5}, epoch+1)
            args.writer.add_scalars(
                'finetune_lr', {'train': optimiser.param_groups[0]['lr']}, epoch+1)

        valid_loss, valid_acc, valid_acc_top5 = evaluate(
            encoder, dataloaders, 'valid', epoch, args)

        # For the best performing epoch, reset patience and save model,
        # else update patience.
        if valid_acc >= best_valid_acc:
            patience_counter = 0
            best_epoch = epoch + 1
            best_valid_acc = valid_acc

            # saving using process (rank) 0 only as all processes are in sync

            state = {
                #'args': args,
                'base_encoder': encoder.state_dict(),
                'optimiser': optimiser.state_dict(),
                'epoch': epoch
            }

            torch.save(state, (args.checkpoint_dir[:-3] + "_finetune.pt"))
        else:
            patience_counter += 1
            if patience_counter == (args.patience - 10):
                logging.info('\nPatience counter {}/{}.'.format(
                    patience_counter, args.patience))
            elif patience_counter == args.patience:
                logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
                    args.patience))
                break

        epoch_finetune_loss = None  # reset loss
        epoch_finetune_acc = None
        epoch_finetune_acc_top5 = None

    del state

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory


def evaluate(encoder, dataloaders, mode, epoch, args):
    ''' Evaluate script - MoCo

        Evaluate the encoder and Linear Evaluation head with Cross Entropy loss.
    '''

    epoch_valid_loss = None  # reset loss
    epoch_valid_acc = None  # reset acc
    epoch_valid_acc_top5 = None

    ''' Loss / Criterion '''
    criterion = nn.CrossEntropyLoss().cuda()

    # initilize Variables
    args.writer = SummaryWriter(args.summaries_dir)

    # Evaluate both encoder and class head
    encoder.eval()

    # initilize Variables
    sample_count = 0
    run_loss = 0
    run_top1 = 0.0
    run_top5 = 0.0

    # Print setup for distributed only printing on one node.
    if args.print_progress:
            # tqdm for process (rank) 0 only when using distributed training
        eval_dataloader = tqdm(dataloaders[mode])
    else:
        eval_dataloader = dataloaders[mode]

    ''' epoch loop '''
    for i, (inputs, target) in enumerate(eval_dataloader):

        # Do not compute gradient for encoder and classification head
        encoder.zero_grad()

        inputs = inputs.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)

        # Forward pass

        output = encoder(inputs)

        loss = criterion(output, target)

        torch.cuda.synchronize()

        sample_count += inputs.size(0)

        run_loss += loss.item()

        predicted = output.argmax(-1)

        acc = (predicted == target).sum().item() / target.size(0)

        run_top1 += acc

        _, output_topk = output.topk(5, 1, True, True)

        acc_top5 = (output_topk == target.view(-1, 1).expand_as(output_topk)
                    ).sum().item() / target.size(0)  # num corrects

        run_top5 += acc_top5

    epoch_valid_loss = run_loss / len(dataloaders[mode])  # sample_count

    epoch_valid_acc = run_top1 / len(dataloaders[mode])

    epoch_valid_acc_top5 = run_top5 / len(dataloaders[mode])

    ''' Printing '''
    if args.print_progress:  # only validate using process 0
        logging.info('\n[{}] loss: {:.4f},\t acc: {:.4f},\t acc_top5: {:.4f} \n'.format(
            mode, epoch_valid_loss, epoch_valid_acc, epoch_valid_acc_top5))

        if mode != 'test':
            args.writer.add_scalars('finetune_epoch_loss', {mode: epoch_valid_loss}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc', {mode: epoch_valid_acc}, epoch+1)
            args.writer.add_scalars('finetune_epoch_acc_top5', {
                                    'train': epoch_valid_acc_top5}, epoch+1)

    torch.cuda.empty_cache()

    gc.collect()  # release unreferenced memory

    return epoch_valid_loss, epoch_valid_acc, epoch_valid_acc_top5
