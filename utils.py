from typing import Literal, Callable

import torch

def create_noise(batch_size: int, z_size: int, mode_z: Literal["uniform", "normal"]):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size, 1, 1)*2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size, 1, 1)
    return input_z


def create_samples(g_model: Callable, input_z: torch.Tensor, batch_size: int, image_size: tuple):
    g_output = g_model(input_z)
    images = torch.reshape(g_output, (batch_size, *image_size))
    return (images+1)/2.0


