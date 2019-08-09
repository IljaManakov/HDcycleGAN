"""module containing configurations for the model and training routine"""

from torch.nn.functional import relu, selu, leaky_relu, tanh
from torch.nn import MSELoss, L1Loss, SmoothL1Loss
from torch.optim import Adam

discriminator = {
    'n_layers': 7,
    'kernel_size': (3, 3),
    'activation': selu,
    'channel_factor': 8,
    'max_channels': 1024,
    'input_channels': 1,
    'n_residual': (0, 0),
    'affine': True
}

dataset = {
    'vmin': 'mean-0.5',
    'vmax': 1.0,
    'whitening': False
}

dataloader = {
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 2,
}

optimizer = {
    'optimizer_type': Adam,
    'learning_rate': 0.005,
    'apex': False,
    'lr_decay': (0.99, 1e5)
}

training = {
    'epochs': 25,
    'name': 'discriminator-test',
}


def get_complete_config():
    import config
    complete_config = {key: value for key, value in config.__dict__.items()
                       if isinstance(value, dict) and key != '__builtins__'}
    return complete_config


