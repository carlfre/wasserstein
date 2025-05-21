from typing import Literal, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from models.vae import VAE
from models.discriminators import DiscriminatorWGAN
# from models.generators import make_generator_network_wgan
from utils import create_noise

from models.load_model import load_vae_model, load_generator_model
from load_data import load_config


class GeneratedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], "place_holder_label" 
    


def generate_dataset_vae(
        vae: VAE,
        dataset_size: int,
        config: dict[str],      
) -> tuple[Dataset, DataLoader]:
    """
    Generate a dataset of samples using the VAE model.
    
    Args:
        vae (VAE): The VAE model to use for generating samples.
        dataset_size (int): The number of samples to generate.
        batch_size (int): The batch size for the DataLoader.
        device (str): The device to use for computation ('cpu' or 'cuda').
        
    Returns:
        tuple: A tuple containing the generated dataset and DataLoader.
    """
    training_config = config["training"]

    device = training_config["device"]
    batch_size = training_config["batch_size"]


    # Generate random latent vectors
    z = torch.randn(dataset_size, vae.latent_dim).to(device)
    with torch.no_grad():
        generated_samples = vae.decoder(z)

    # if len(generated_samples.shape) == 4:
    #     generated_samples = generated_samples[:, 0, :, :]
    print(generated_samples.shape)

    dataset = GeneratedDataset(generated_samples.cpu())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataset, dataloader




def generate_dataset_wgan(
    generator,
    dataset_size: int,
    config: dict[str],
) -> tuple[Dataset, DataLoader]:
    """
    Generate a dataset of samples using the WGAN generator.
    
    Args:
    generator: The WGAN generator model to use for generating samples.
    dataset_size (int): The number of samples to generate.
    batch_size (int): The batch size for the DataLoader.
    latent_dim (int): The dimension of the latent space.
    device (str): The device to use for computation ('cpu' or 'cuda').
    
    Returns:
    tuple: A tuple containing the generated dataset and DataLoader.
    """
    training_config = config["training"]
    model_config = config["model_specifics"]

    device = training_config["device"]
    batch_size = training_config["batch_size"]
    latent_dim = model_config["latent_dim"]
    latent_distribution = model_config["latent_distribution"]

    # Generate random latent vectors
    # z = torch.randn(dataset_size, latent_dim).to(device)
    z = create_noise(dataset_size, latent_dim, latent_distribution).to(device)
    
    with torch.no_grad():
        generator.eval()
        generated_samples = generator(z)
    
    # print(generated_samples.shape)
    # Move the generated samples to CPU
    dataset = GeneratedDataset(generated_samples.cpu())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataset, dataloader


def generate_data(model_type: Literal["vae", "wgan"], gen_nr: int, n_datapoints, label: str = "experiment_4", device: Optional[str] = None) -> list[torch.Tensor]:
    if  model_type == "vae":
        config = load_config("configs/vae_config.yaml")
        vae_path = f"checkpoints/gen_{gen_nr}_vae_{label}.pth"
        vae = load_vae_model(config, vae_path)
        dataset, _ = generate_dataset_vae(vae, n_datapoints, config)
    elif model_type == "wgan":
        config = load_config("configs/wgan_config.yaml")
        generator_path = f"checkpoints/gen_{gen_nr}_wgan_generator_{label}.pth"
        generator = load_generator_model(config, generator_path)
        dataset, _ = generate_dataset_wgan(generator, n_datapoints, config)
    else:
        raise ValueError("Invalid model type. Choose 'vae' or 'wgan'.")

    images = [sample[0] for sample in dataset]
    if device is not None:
        images = [img.to(device) for img in images]
    return images