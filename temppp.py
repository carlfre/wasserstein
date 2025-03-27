import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from loss_functions.wgan_loss import gradient_penalty
from models.generators import make_generator_network_wgan
from models.discriminators import DiscriminatorWGAN
from utils import create_noise, create_samples
from load_data import load_mnist



# z_size = 100
# n_filters = 64
# device = 'cuda'

# lambda_gp = 10
# batch_size = 64
# mode_z = 'uniform'

# mnist_dataset = load_mnist()
# mnist_dl = DataLoader(
#     mnist_dataset, batch_size=batch_size,
#     shuffle=True, drop_last=True
# )


# for x, _ in mnist_dl:
#     print(x.shape)
#     break


import yaml

with open('wgan_config.yaml') as f:
    config = yaml.safe_load(f)
    print(config)