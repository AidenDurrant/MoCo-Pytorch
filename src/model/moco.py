# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.network as models


class MoCo_Model(nn.Module):
    def __init__(self, args, queue_size=65536, momentum=0.999, temperature=0.07):
        '''
        MoCoV2 model, taken from: https://github.com/facebookresearch/moco.

        Adapted for use in personal Boilerplate for unsupervised/self-supervised contrastive learning.

        Additionally, too inspiration from: https://github.com/HobbitLong/CMC.

        Args:
            init:
                args (dict): Program arguments/commandline arguments.

                queue_size (int): Length of the queue/memory, number of samples to store in memory. (default: 65536)

                momentum (float): Momentum value for updating the key_encoder. (default: 0.999)

                temperature (float): Temperature used in the InfoNCE / NT_Xent contrastive losses. (default: 0.07)

            forward:
                x_q (Tensor): Reprentation of view intended for the query_encoder.

                x_k (Tensor): Reprentation of view intended for the key_encoder.

        returns:

            logit (Tensor): Positve and negative logits computed as by InfoNCE loss. (bsz, queue_size + 1)

            label (Tensor): Labels of the positve and negative logits to be used in softmax cross entropy. (bsz, 1)

        '''
        super(MoCo_Model, self).__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        assert self.queue_size % args.batch_size == 0  # for simplicity

        # Load model
        self.encoder_q = getattr(models, args.model)(
            args, num_classes=128)  # Query Encoder

        self.encoder_k = getattr(models, args.model)(
            args, num_classes=128)  # Key Encoder

        # Add the mlp head
        self.encoder_q.fc = models.projection_MLP(args)
        self.encoder_k.fc = models.projection_MLP(args)

        # Initialize the key encoder to have the same values as query encoder
        # Do not update the key encoder via gradient
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue to store negative samples
        self.register_buffer("queue", torch.randn(self.queue_size, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        '''
        Update the key_encoder parameters through the momentum update:


        key_params = momentum * key_params + (1 - momentum) * query_params

        '''

        # For each of the parameters in each encoder
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)

    @torch.no_grad()
    def shuffled_idx(self, batch_size):
        '''
        Generation of the shuffled indexes for the implementation of ShuffleBN.

        https://github.com/HobbitLong/CMC.

        args:
            batch_size (Tensor.int()):  Number of samples in a batch

        returns:
            shuffled_idxs (Tensor.long()): A random permutation index order for the shuffling of the current minibatch

            reverse_idxs (Tensor.long()): A reverse of the random permutation index order for the shuffling of the
                                            current minibatch to get back original sample order

        '''

        # Generate shuffled indexes
        shuffled_idxs = torch.randperm(batch_size).long().cuda()

        reverse_idxs = torch.zeros(batch_size).long().cuda()

        value = torch.arange(batch_size).long().cuda()

        reverse_idxs.index_copy_(0, shuffled_idxs, value)

        return shuffled_idxs, reverse_idxs

    @torch.no_grad()
    def update_queue(self, feat_k):
        '''
        Update the memory / queue.

        Add batch to end of most recent sample index and remove the oldest samples in the queue.

        Store location of most recent sample index (ptr).

        Taken from: https://github.com/facebookresearch/moco

        args:
            feat_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.
        '''

        batch_size = feat_k.size(0)

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = feat_k

        # move pointer along to end of current batch
        ptr = (ptr + batch_size) % self.queue_size

        # Store queue pointer as register_buffer
        self.queue_ptr[0] = ptr

    def InfoNCE_logits(self, f_q, f_k):
        '''
        Compute the similarity logits between positive
         samples and positve to all negatives in the memory.

        args:
            f_q (Tensor): Feature reprentations of the view x_q computed by the query_encoder.

            f_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.

        returns:
            logit (Tensor): Positve and negative logits computed as by InfoNCE loss. (bsz, queue_size + 1)

            label (Tensor): Labels of the positve and negative logits to be used in softmax cross entropy. (bsz, 1)
        '''

        f_k = f_k.detach()

        # Get queue from register_buffer
        f_mem = self.queue.clone().detach()

        # Normalize the feature representations
        f_q = nn.functional.normalize(f_q, dim=1)
        f_k = nn.functional.normalize(f_k, dim=1)
        f_mem = nn.functional.normalize(f_mem, dim=1)

        # Compute sim between positive views
        pos = torch.bmm(f_q.view(f_q.size(0), 1, -1),
                        f_k.view(f_k.size(0), -1, 1)).squeeze(-1)

        # Compute sim between postive and all negatives in the memory
        neg = torch.mm(f_q, f_mem.transpose(1, 0))

        logits = torch.cat((pos, neg), dim=1)

        logits /= self.temperature

        # Create labels, first logit is postive, all others are negative
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels

    def forward(self, x_q, x_k):

        batch_size = x_q.size(0)

        # Feature representations of the query view from the query encoder
        feat_q = self.encoder_q(x_q)

        # TODO: shuffle ids with distributed data parallel
        # Get shuffled and reversed indexes for the current minibatch
        shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size)

        with torch.no_grad():
            # Update the key encoder
            self.momentum_update()

            # Shuffle minibatch
            x_k = x_k[shuffled_idxs]

            # Feature representations of the shuffled key view from the key encoder
            feat_k = self.encoder_k(x_k)

            # reverse the shuffled samples to original position
            feat_k = feat_k[reverse_idxs]

        # Compute the logits for the InfoNCE contrastive loss.
        logit, label = self.InfoNCE_logits(feat_q, feat_k)

        # Update the queue/memory with the current key_encoder minibatch.
        self.update_queue(feat_k)

        return logit, label
