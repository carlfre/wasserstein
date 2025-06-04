from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def plot_image(image: Union[torch.Tensor, np.ndarray], filename=None, show=True) -> None:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if len(image.shape) == 3:
        if image.shape[0] != 1:
            raise ValueError("Image is 3d for some reason?")

        image = image[0]

    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()


def plot_grid(images: list[torch.Tensor], n_cols: int = 5) -> None:
    # images = [(im - im.min()) / (im.max() - im.min()) for im in images]
    grid = make_grid(images, nrow=n_cols, padding=0)

    grid_np = grid.permute(1, 2, 0).numpy()

    # # Apply min-max scaling before plotting.
    # if grid_np.max() != grid_np.min():
    #     grid_np = (grid_np - grid_np.min()) / (grid_np.max() - grid_np.min())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.imshow(grid_np)