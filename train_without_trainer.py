import importlib
import argparse
import os
from os.path import join, isdir
from itertools import chain

import torch as pt
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
import numpy as np

from optimizer import OptimizerCollection


def load_config(file):
    """
    initialize module from .py config file
    :param file: filename of the config .py file
    :return: config as module
    """

    spec = importlib.util.spec_from_file_location("config", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def set_seed(seed):
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False
    np.random.seed(seed)
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """

    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def backward(loss):
    loss.backward()


# cmd arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', required=True, dest='config')
parser.add_argument('--n_epochs', '-n', default=50, type=int, dest='n_epochs')
parser.add_argument('--logdir', '-l', default='./', dest='logdir')
parser.add_argument('--seed', '-s', default=0, type=int, dest='seed')
args = parser.parse_args()

# fix seed and load config
set_seed(args.seed)
config = load_config(args.config)

# use fp16 if possible
dtype = pt.float16 if hasattr(config, 'APEX') else pt.float32

# initialize components
model = config.MODEL(**config.model).cuda().to(dtype)
model.apply(weight_init)
dataset = config.DATASET(**config.dataset)
criterion = config.LOSS(**config.loss)
gen_optimizer = config.OPTIMIZER(chain(model.generator['hn'].parameters(),
                                       model.generator['ln'].parameters()) , **config.optimizer)
disc_optimizer = config.OPTIMIZER(model.discriminator.parameters(), **config.optimizer)
optimizers = []
for optimizer in (disc_optimizer, gen_optimizer):
    if hasattr(config, 'APEX'):
        optimizer = config.APEX(optimizer, **config.apex)
    optimizers.append(optimizer)
optimizer = OptimizerCollection(*optimizers)
dataloader = DataLoader(dataset, **config.dataloader)

# convert optimizer and unify backward call
if dtype == pt.float16:
    optimizer = config.APEX(optimizer, **config.apex)
if not hasattr(optimizer, 'backward'):
    setattr(optimizer, 'backward', backward)

# init log directory
if not isdir(args.logdir):
    os.makedirs(args.logdir)

# training loop
steps_in_epoch = len(dataloader)
try:
    for epoch in range(args.n_epochs):
        for step, (images, labels) in enumerate(dataloader):

                # convert data to match network
                images = [i.cuda().to(dtype) for i in images]
                labels = [l.cuda().long() for l in labels]

                # forward pass
                result = model(images)
                loss = criterion(result, ([i.float() for i in images], labels))

                # backward pass
                optimizer.zero_grad()
                optimizer.backward(loss)
                optimizer.step()

                # reporting
                progress = (epoch*steps_in_epoch+step) / (args.n_epochs*steps_in_epoch)
                progress = round(100*progress, 2)
                print(f'\repoch: {epoch} of {args.n_epochs}, step: {step}, progress: {progress}%,'
                      f' loss: {round(loss.item(), 4)}', end='')

                # write loss
                with open(join(args.logdir, 'losses.csv'), 'a') as file:
                    print(epoch, step, loss, sep=',', file=file)

# always save model and optimizer before exiting
finally:
    pt.save(model.state_dict(), join(args.logdir, f'{model.__class__.__name__}_epoch-{epoch}_step-{step}.pt'))
    pt.save(optimizer.state_dict(), join(args.logdir, f'{optimizer.__class__.__name__}_epoch-{epoch}_step-{step}.pt'))
