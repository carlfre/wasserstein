from typing import Literal

import matplotlib.pyplot as plt
import torch
import numpy as np

from generate_dataset import generate_data
from load_data import load_mnist, load_config


plt.rcParams.update({'font.size': 20})

def plot_pixel_intensities(model_type: Literal["vae", "wgan"], gen_nrs: list[int] | int, n_images: int = 10_000) -> None:
    if isinstance(gen_nrs, int):
        gen_nrs = [gen_nrs]
    
    intensities_list = []
    for gen_nr in gen_nrs:
        images = generate_data(model_type, gen_nr, n_images)

        images = torch.stack(images)

        images = images.view(images.size(0), -1)  # Flatten the images
        images = images.cpu().numpy()  # Convert to numpy array


        intensities = images.reshape(-1)
        intensities_list.append(intensities)


    plt.figure(figsize=(10, 5))
    for intensities, gen_nr in zip(intensities_list, gen_nrs):
        counts, bin_edges = np.histogram(intensities, bins=200, density=True)
        log_density = np.log1p(counts)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.plot(bin_centers, log_density, label=f"Generation {gen_nr + 1}", alpha=1)
        # plt.bar(bin_centers, log_density, width=bin_edges[1] - bin_edges[0], alpha=0.3)  # , label=f"Generation {gen_nr + 1}", color='blue')


        # plt.hist(intensities, bins=200, alpha=0.3, label=f"Generation {gen_nr + 1}", density=True) # add 1 to gen_nr to get 1-indexing.

    plt.xlabel("Pixel Intensity")
    plt.ylabel("log(1 + Density)")
    plt.legend()
    plt.xlim(-0.2,1.1)
    plt.savefig(f"plots/pixel_intensity_{model_type}.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_mnist_pixel_intensities(config: dict, n_images: int = 1_000) -> None:
    train_loader, test_loader, train_set, test_set = load_mnist(config)

    # Extract pixel intensities
    intensities = []
    for i, (img, _) in enumerate(train_loader):
        if i >= n_images:
            break
        intensities.append(img.view(-1).numpy())
        

    intensities = np.concatenate(intensities)
    counts, bin_edges = np.histogram(intensities, bins=200, density=True)
    log_density = np.log1p(counts)

    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Plot
    plt.figure(figsize=(10, 5))
    # plt.bar(bin_centers, log_density, width=bin_edges[1] - bin_edges[0], alpha=0.5, label="MNIST", color='blue')
    plt.plot(bin_centers, log_density, label="MNIST", color='blue')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("log(1 + Density)")
    plt.legend()
    plt.xlim(-0.2, 1.1)
    plt.savefig("plots/pixel_intensity_mnist.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

def main():
    n_images = 20_000
    gens = [0, 10, 19]
    plot_pixel_intensities("vae", gens, n_images)
    plot_pixel_intensities("wgan", gens, n_images)
    config = load_config("configs/vae_config.yaml")
    plot_mnist_pixel_intensities(config, n_images)


if __name__ == "__main__":
    main()


