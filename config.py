"""module containing configurations for the model and training routine"""

from torch.nn.functional import relu
from torch.nn import L1Loss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch import float16

from model import Discriminator, HDCycleGAN, Generator, CycleGAN
from dataset import OCTQualityDataset
from critertion import ClassificationLoss, ClassificationLossHD


dtype = float16
cuda = True
seed = 0

MODEL = HDCycleGAN
LOSS = ClassificationLossHD
DATASET = OCTQualityDataset
OPTIMIZER = Adam
LOGDIR = f'./trained/{seed}'

try:
    from apex.fp16_utils import FP16_Optimizer
    APEX = FP16_Optimizer
    apex = {
        'dynamic_loss_scale': True,
        'dynamic_loss_args': {'init_scale': 2 ** 10},
        'verbose': False
    }
except ImportError:
    pass

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
    'parent_folder': '/home/ilja/Datasets/oct_quality',  # replace with the path to the parent folder on your system
    'fraction': 0.1
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

trainer = {
    'loss_decay': 0.8,
    'split_sample': lambda x: (x[0], x)
}

