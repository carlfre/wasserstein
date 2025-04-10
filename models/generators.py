import torch
import torch.nn as nn

def make_generator_network_wgan(latent_dim: int, n_channel_scaling_factor: int):
    model = nn.Sequential(
    nn.ConvTranspose2d(latent_dim, n_channel_scaling_factor*4, 4,
    1, 0, bias=False),
    nn.InstanceNorm2d(n_channel_scaling_factor*4),
    nn.LeakyReLU(0.2),

    nn.ConvTranspose2d(n_channel_scaling_factor*4, n_channel_scaling_factor*2,
    3, 2, 1, bias=False),
    nn.InstanceNorm2d(n_channel_scaling_factor*2),
    nn.LeakyReLU(0.2),

    nn.ConvTranspose2d(n_channel_scaling_factor*2, n_channel_scaling_factor, 4,
    2, 1, bias=False),
    nn.InstanceNorm2d(n_channel_scaling_factor),
    nn.LeakyReLU(0.2),

    nn.ConvTranspose2d(n_channel_scaling_factor, 1, 4, 2, 1, bias=False),
    nn.Tanh()
    )
    return model