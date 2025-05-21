from typing import Literal

import torch
import numpy as np

from load_data import load_config, load_mnist
from generate_dataset import generate_data
from models.load_model import load_discriminator_model




def discriminate_images(model_type: Literal["vae", "wgan"], gen_nr: int, n_images: int = 10_000, discriminator_path: str = "checkpoints/gen_0_wgan_discriminator_experiment_4.pth") -> None:
    wgan_config = load_config("configs/wgan_config.yaml")
    device = wgan_config["training"]["device"]
    discriminator = load_discriminator_model(wgan_config, discriminator_path)

    images = generate_data(model_type, gen_nr, n_images, device=device)
    classifications = np.array([discriminator(im).cpu().detach().numpy() for im in images]).flatten()

    correctly_classified = np.sum(classifications <= 0.5)
    incorrectly_classified = np.sum(classifications > 0.5)
    classification_rate = correctly_classified / (correctly_classified + incorrectly_classified)
    # print(classifications)
    return classification_rate
    # return np.mean(classifications)


    
def classification_rate_mnist(discriminator_path: str = "checkpoints/gen_0_wgan_discriminator_experiment_4.pth"):
    wgan_config = load_config("configs/wgan_config.yaml")
    device = wgan_config["training"]["device"]
    discriminator = load_discriminator_model(wgan_config, discriminator_path)

    train_loader, test_loader, train_set, test_set = load_mnist(wgan_config)

    images = train_set.data[:1_000].unsqueeze(1).float().to(device) / 255
    
    classifications = np.array([discriminator(im).cpu().detach().numpy() for im in images]).flatten()
    correctly_classified = np.sum(classifications > 0.5)
    incorrectly_classified = np.sum(classifications <= 0.5)
    classification_rate = correctly_classified / (correctly_classified + incorrectly_classified)
    print(f"Classification rate: {classification_rate}")
    return classification_rate




classification_rate_mnist()    

# rates = [discriminate_images("vae", i, 1_000) for i in range(20)]
# print(rates)

