# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required


def get_optimiser(models, mode, args):
    '''Get the desired optimiser

    - Selects and initialises an optimiser with model params.

    - if 'LARS' is selected, the 'bn' and 'bias' parameters are removed from
     model optimisation, only passing the parameters we want.

    Args:
        models (tuple): models which we want to optmise, (e.g. encoder and projection head)

        mode (string): the mode of training, (i.e. 'pretrain', 'finetune')

        args (Dictionary): Program Arguments

    Returns:
        optimiser (torch.optim.optimizer):
    '''
    params_models = []
    reduced_params = []

    removed_params = []

    skip_lists = ['bn', 'bias']

    for m in models:

        m_skip = []
        m_noskip = []

        params_models += list(m.parameters())

        for name, param in m.named_parameters():
            if (any(skip_name in name for skip_name in skip_lists)):
                m_skip.append(param)
            else:
                m_noskip.append(param)
        reduced_params += list(m_noskip)
        removed_params += list(m_skip)
    # Set hyperparams depending on mode
    if mode == 'pretrain':
        lr = args.learning_rate
        wd = args.weight_decay
    else:
        lr = args.finetune_learning_rate
        wd = args.finetune_weight_decay

    # Select Optimiser
    if args.optimiser == 'adam':

        optimiser = optim.Adam(params_models, lr=lr,
                               weight_decay=wd)

    elif args.optimiser == 'sgd':

        optimiser = optim.SGD(params_models, lr=lr,
                              weight_decay=wd, momentum=0.9)

    elif args.optimiser == 'lars':

        print("reduced_params len: {}".format(len(reduced_params)))
        print("removed_params len: {}".format(len(removed_params)))

        optimiser = LARS(reduced_params+removed_params, lr=lr,
                         weight_decay=wd, eta=0.001, use_nesterov=False, len_reduced=len(reduced_params))
    else:

        raise NotImplementedError('{} not setup.'.format(args.optimiser))

    return optimiser


class LARS(Optimizer):
    """
    Layer-wise adaptive rate scaling

    - Converted from Tensorflow to Pytorch from:

    https://github.com/google-research/simclr/blob/master/lars_optimizer.py

    - Based on:

    https://github.com/noahgolmant/pytorch-lars

    params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)

        lr (int): Length / Number of layers we want to apply weight decay, else do not compute

        momentum (float, optional): momentum factor (default: 0.9)

        use_nesterov (bool, optional): flag to use nesterov momentum (default: False)

        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
            ("\beta")

        eta (float, optional): LARS coefficient (default: 0.001)

    - Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.

    - Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888

    """

    def __init__(self, params, lr, len_reduced, momentum=0.9, use_nesterov=False, weight_decay=0.0, classic_momentum=True, eta=0.001):

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            classic_momentum=classic_momentum,
            eta=eta,
            len_reduced=len_reduced
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eta = eta
        self.len_reduced = len_reduced

    def step(self, epoch=None, closure=None):

        loss = None

        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            learning_rate = group['lr']

            # TODO: Hacky
            counter = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: This really hacky way needs to be improved.

                # Note Excluded are passed at the end of the list to are ignored
                if counter < self.len_reduced:
                    grad += self.weight_decay * param

                # Create parameter for the momentum
                if "momentum_var" not in param_state:
                    next_v = param_state["momentum_var"] = torch.zeros_like(
                        p.data
                    )
                else:
                    next_v = param_state["momentum_var"]

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: implementation of layer adaptation
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()

                    trust_ratio = torch.where(w_norm.ge(0), torch.where(
                        g_norm.ge(0), (self.eta * w_norm / g_norm), torch.Tensor([1.0]).to(device)), torch.Tensor([1.0]).to(device)).item()

                    scaled_lr = learning_rate * trust_ratio

                    next_v.mul_(momentum).add_(scaled_lr, grad)

                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)

                # Not classic_momentum
                else:

                    next_v.mul_(momentum).add_(grad)

                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (grad)

                    else:
                        update = next_v

                    trust_ratio = 1.0

                    # TODO: implementation of layer adaptation
                    w_norm = torch.norm(param)
                    v_norm = torch.norm(update)

                    device = v_norm.get_device()

                    trust_ratio = torch.where(w_norm.ge(0), torch.where(
                        v_norm.ge(0), (self.eta * w_norm / v_norm), torch.Tensor([1.0]).to(device)), torch.Tensor([1.0]).to(device)).item()

                    scaled_lr = learning_rate * trust_ratio

                    p.data.add_(-scaled_lr * update)

                counter += 1

        return loss
