import torch
import torch.nn as nn

def make_generator_network_wgan(size_of_noise_vector: int, n_filters: int):
    model = nn.Sequential(
    nn.ConvTranspose2d(size_of_noise_vector, n_filters*4, 4,
    1, 0, bias=False),
    nn.InstanceNorm2d(n_filters*4),
    nn.LeakyReLU(0.2),

    nn.ConvTranspose2d(n_filters*4, n_filters*2,
    3, 2, 1, bias=False),
    nn.InstanceNorm2d(n_filters*2),
    nn.LeakyReLU(0.2),

    nn.ConvTranspose2d(n_filters*2, n_filters, 4,
    2, 1, bias=False),
    nn.InstanceNorm2d(n_filters),
    nn.LeakyReLU(0.2),

    nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
    nn.Tanh()
    )
    return model