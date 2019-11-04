"""module containing configurations for the model and training routine"""

from torch.nn.functional import relu
from torch.nn import L1Loss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch import float16

from apex.fp16_utils import FP16_Optimizer

from model import Discriminator, HDCycleGAN, Generator, CycleGAN
from dataset import OCTQualityDatasetHDF5
from critertion import ClassificationLoss, ClassificationLossHD


dtype = float16
cuda = True
seed = 0

MODEL = HDCycleGAN
LOSS = ClassificationLossHD
DATASET = OCTQualityDatasetHDF5
OPTIMIZER = Adam
APEX = FP16_Optimizer
LOGDIR = f'./trained/{seed}'

generator = {
    'scale_factor': 3,
    'channel_factor': 16,
    'activation': relu,
    'kernel_size': (3, 3),
    'n_residual': (3, 3),
    'input_channels': 1,
    'skip_conn': 'concat'
}

discriminator = {
    'n_layers': 7,
    'kernel_size': (3, 3),
    'activation': relu,
    'channel_factor': 16,
    'max_channels': 1024,
    'input_channels': 1,
    'n_residual': (0, 0),
    'affine': False
}

model = {
    'discriminator': (Discriminator, discriminator),
    'generator': (Generator, generator),
    'input_size': (1, 512, 512),
    'pool_size': 32,
    'pool_write_probability': 1
}

dataset = {
    'storage_dir': '/home/ilja/Datasets/',  # enter the directory of your hdf5 files here
    'hn': 'low.hdf5',  # enter the name of the HN hdf5 file here
    'ln': 'high.hdf5',  # enter the name of the LN hdf5 file here
    'vmin': 'mean-0.5',
    'vmax': 1.0,
    'whitening': False
}

dataloader = {
    'batch_size': 4,
    'shuffle': True,
    'num_workers': 0,
}

loss = {
    'cycle_loss': L1Loss,
    'discriminator_loss': CrossEntropyLoss,
    'cycle_factor': 10
}

optimizer = {
    'lr': 0.0005
}

apex = {
    'dynamic_loss_scale': True,
    'dynamic_loss_args': {'init_scale': 2**16},
    'verbose': False
}

trainer = {
    'loss_decay': 0.8,
    'split_sample': lambda x: (x[0], x)
}

