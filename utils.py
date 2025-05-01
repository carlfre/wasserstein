from typing import Literal, Callable
import csv

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




def write_list_to_csv(
    content: list,
    file_path: str
) -> None:
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(content)

def read_list_from_csv(
    file_path: str
) -> list:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        content = next(reader)
    return [float(i) for i in content]


