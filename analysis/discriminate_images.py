from typing import Literal

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from load_data import load_config, load_mnist
from generate_dataset import generate_data
from models.load_model import load_discriminator_model

sns.set_theme(palette="deep", font_scale=1.3)

def classification_rate(
        realness_scores: np.ndarray,
        image_type: Literal["real", "generated"],
        threshold: float = 0.5, 
) -> float:
    """
    Calculate the rate of correct classifications.
    
    Args:
        realness_scores (list[float]): List of realness scores for images.
        image_type (Literal["real", "generated"]): Type of images, either "real" or "generated".
        threshold (float): Threshold to classify images as real or generated.
        
    Returns:
        float: Classification rate.
    """
    n = realness_scores.shape[0]
    n_classified_as_real = np.sum(np.array(realness_scores) > threshold)

    if image_type == "real":
        return n_classified_as_real / n
    elif image_type == "generated":
        return (n - n_classified_as_real) / n
    else:
        raise ValueError("image_type must be either 'real' or 'generated'")


def discriminate_generated_images(
    model_type: Literal["vae", "wgan"],
    gen_nr: int,
    discriminator_gen: int,
    discriminator_label: str,
    generator_label: str = "experiment_4",
    n_images: int = 10_000,
) -> list[float]:
    discriminator_path = f"checkpoints/gen_{discriminator_gen}_wgan_discriminator_{discriminator_label}.pth"

    wgan_config = load_config("configs/wgan_config.yaml")
    device = wgan_config["training"]["device"]
    discriminator = load_discriminator_model(wgan_config, discriminator_path)

    images = generate_data(model_type, gen_nr, n_images, label=generator_label, device=device)
    realness_scores = np.array(
        [discriminator(im).cpu().detach().numpy() for im in images]
    ).flatten()
    return realness_scores



def discriminate_mnist_images(
    discriminator_gen: int,
    discriminator_label: str,
    n_images: int = 1_000
):
    discriminator_path = f"checkpoints/gen_{discriminator_gen}_wgan_discriminator_{discriminator_label}.pth"
    wgan_config = load_config("configs/wgan_config.yaml")
    device = wgan_config["training"]["device"]
    discriminator = load_discriminator_model(wgan_config, discriminator_path)

    train_loader, test_loader, train_set, test_set = load_mnist(wgan_config)

    images = test_set.data[:n_images].unsqueeze(1).float().to(device) / 255

    realness_scores = np.array(
        [discriminator(im).cpu().detach().numpy() for im in images]
    ).flatten()
    return realness_scores


def mnist_classification_rate_plot():
    gens = list(range(20))

    for setup_nr, label in enumerate(["experiment_4", "experiment_5"]):
        classification_rates = []
        for gen_nr in gens:
            print("gen_nr", gen_nr)
            realness_scores = discriminate_mnist_images(
                discriminator_gen=gen_nr,
                discriminator_label=label,
                n_images=10_000
            )
            classification_rates.append(classification_rate(realness_scores, "real"))

        plt.plot(
            gens,
            classification_rates,
            label=f"Setup {setup_nr + 1}",
            marker="o",
            markersize=8,
        )

    plt.xlabel("Generation")
    plt.ylabel("Classification Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/mnist_classification_rate.pdf")


def generated_image_classification_rate_heatmap_plot(
    generator_type: Literal["vae", "wgan"],
    discriminator_label: str,
    generator_label: str,
    n_images: int = 1_000,
):
    n_gens = 20
    classification_scores = np.zeros((n_gens, n_gens))

    for generator_gen in range(n_gens):
        for discriminator_gen in range(n_gens):
            print(f"Generator Gen: {generator_gen}, Discriminator Gen: {discriminator_gen}")
            realness_scores = discriminate_generated_images(
                model_type=generator_type,
                gen_nr=generator_gen,
                discriminator_gen=discriminator_gen,
                discriminator_label=discriminator_label,
                generator_label=generator_label,
                n_images=n_images
            )
            classification_scores[generator_gen, discriminator_gen] = classification_rate(
                realness_scores, "generated"
            )

    ax = sns.heatmap(
        classification_scores,
        fmt=".2f",
    )
    
    ticks = np.arange(0, n_gens, 2) + 0.5 # +0.5 to center ticks
    tick_labels = list(range(1, n_gens + 1, 2))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    plt.xlabel("Discriminator Generation")
    plt.ylabel("Generator Generation")
    plt.tight_layout()
    plt.savefig(f"plots/generator_{generator_type}_{generator_label}_discriminator_{discriminator_label}_classification_rate_heatmap.pdf")
    plt.show()



n_images = 500

# generator_type = "vae"
# for discriminator_label in ["experiment_4", "experiment_5"]:
#     print(f"Generating heatmap for {generator_type} with {discriminator_label}")
#     generated_image_classification_rate_heatmap_plot(
#         generator_type,
#         discriminator_label=discriminator_label,
#         generator_label="experiment_4",
#         n_images=n_images
#     )


generator_type = "wgan"
for discriminator_label in ["experiment_4", "experiment_5"]:
    print(f"Generating heatmap for {generator_type} with {discriminator_label}")
    generated_image_classification_rate_heatmap_plot(
        generator_type,
        discriminator_label=discriminator_label,
        generator_label="experiment_4",
        n_images=n_images
    )


# generator_type = "wgan"
# for discriminator_label in ["experiment_4", "experiment_5"]:
#     print(f"Generating heatmap for {generator_type} with {discriminator_label}")
#     generated_image_classification_rate_heatmap_plot(
#         generator_type,
#         discriminator_label=discriminator_label,
#         generator_label="experiment_5",
#         n_images=n_images
#     )
