from typing import Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from models.load_model import load_vae_model, load_generator_model
from generate_dataset import generate_dataset_vae, generate_dataset_wgan
from load_data import load_config


#TODO: write some code for plotting an image nicely


def generate_images_vae(
        weight_path: str,
        config: dict[str],
        n_images: int
) -> list[torch.Tensor]:
    vae = load_vae_model(config, weight_path)
    dataset, _ = generate_dataset_vae(vae, n_images, config)
    return [sample[0] for sample in dataset]

def generate_images_wgan(
        generator_weight_path: str,
        config: dict[str],
        n_images: int
) -> list[torch.Tensor]:
    generator = load_generator_model(config, generator_weight_path)
    dataset, _ = generate_dataset_wgan(generator, n_images, config)
    return [sample[0] for sample in dataset]


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
    
    # Apply min-max scaling before plotting.
    if grid_np.max() != grid_np.min():
        grid_np = (grid_np - grid_np.min()) / (grid_np.max() - grid_np.min())
        
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.imshow(grid_np)
    
    
# TODO: Write some code for plotting a grid of images


# if __name__ == "__main__":


#     config = load_config("configs/vae_config.yaml")
#     model_placement = "checkpoints/gen_2_vae_model.pth"

#     vae = load_vae_model(
#         config,
#         model_placement,
#     )

#     dataset, dataloader = generate_dataset_vae(vae, 10, config)

#     # Plot an image from the dataset
#     sample_index = 0  # Choose the first sample
#     sample = dataset[sample_index]
#     image = sample[0]  # Get the image (assuming dataset returns (image, label) pairs)
#     plot_image(image, filename="sample_image.png")


def main():
    run_label = "experiment_4"

    n_images = 20
    plotting_generations = [0, 4, 9, 19] #1-indexed, these are generations 1, 5, 10, 20.

    for gen_nr in plotting_generations:
        generator_path = f"checkpoints/gen_{gen_nr}_wgan_generator_{run_label}.pth"
        config = load_config("configs/wgan_config.yaml")
        images = generate_images_wgan(generator_path, config, n_images)
        plot_grid(images)
        # plt.show()
        plt.savefig(f"plots/generated_images_wgan_gen_{gen_nr}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()

        vae_path = f"checkpoints/gen_{gen_nr}_vae_{run_label}.pth"
        config = load_config("configs/vae_config.yaml")
        images = generate_images_vae(vae_path, config, n_images)
        plot_grid(images)
        plt.savefig(f"plots/generated_images_vae_gen_{gen_nr}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()
        # vae = load_vae_model(config, vae_path)
        # dataset, _ = generate_dataset_vae(vae, n_images, config)
        # images = [sample[0] for sample in dataset]



        # for i, image in enumerate(images):
        #     plot_image(image, filename=f"generated_image_{i}.png", show=False)
        #     plt.show()


if __name__ == "__main__":
    main()
    # Example usage

    # gen_nr = 0
    # generator_path = f"checkpoints/gen_{gen_nr}_wgan_generator_experiment_4.pth"
    # config = load_config("configs/wgan_config.yaml")
    # n_images = 20
    # images = generate_images_wgan(generator_path, config, n_images)
    # plot_grid(images)
    # plt.show()

    # for i, image in enumerate(images):
    #     plot_image(image, filename=f"generated_image_{i}.png", show=False)
    #     plt.show()
