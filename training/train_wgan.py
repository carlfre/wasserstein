import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Literal
from loss_functions.wgan_loss import gradient_penalty
from models.discriminators import DiscriminatorWGAN
from utils import create_noise
import numpy as np


def generator_training_iteration(
    x: torch.Tensor,
    generator: torch.nn.Module,
    discriminator: DiscriminatorWGAN,
    g_optimizer: torch.optim.Optimizer,
    latent_dim: int,
    latent_distribution: Literal["uniform", "normal"],
    device: str,
):
    generator.zero_grad()
    batch_size = x.size(0)
    input_z = create_noise(batch_size, latent_dim, latent_distribution).to(device)
    g_output = generator(input_z)

    d_generated = discriminator(g_output)
    g_loss = -d_generated.mean()

    # gradient backprop & optimize ONLY G's parameters
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()


def discriminator_training_iteration(
    x: torch.Tensor,
    discriminator: DiscriminatorWGAN,
    generator: torch.nn.Module,
    d_optimizer: torch.optim.Optimizer,
    latent_dim: int,
    latent_distribution: str,
    lambda_gp: float,
    device: str,
) -> float:
    discriminator.zero_grad()

    batch_size = x.size(0)
    x = x.to(device)

    # Calculate probabilities on real and generated data
    d_real = discriminator(x)
    input_z = create_noise(batch_size, latent_dim, latent_distribution).to(device)
    g_output = generator(input_z)
    d_generated = discriminator(g_output)
    d_loss = (
        d_generated.mean()
        - d_real.mean()
        + gradient_penalty(x.data, g_output.data, lambda_gp, discriminator, device)
    )
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item()


def wgan_train_epoch(
    models: tuple[torch.nn.Module],
    optimizers: tuple[Optimizer],
    data_loader: DataLoader,
    config: dict,
) -> tuple[list[float], list[float]]:
    generator, discriminator = models
    g_optimizer, d_optimizer = optimizers

    training_config = config["training"]
    model_config = config["model_specifics"]

    device = training_config["device"]

    lambda_gp = model_config["lambda_gp"]
    latent_distribution = model_config["latent_distribution"]
    latent_dim = model_config["latent_dim"]
    n_critic_iterations = model_config["n_critic_iterations"]

    generator.train()
    d_losses, g_losses = [], []
    for i, (x, _) in enumerate(data_loader):
        for _ in range(n_critic_iterations):
            d_loss = discriminator_training_iteration(
                x,
                discriminator,
                generator,
                d_optimizer,
                latent_dim,
                latent_distribution,
                lambda_gp,
                device,
            )
        d_losses.append(d_loss)
        g_losses.append(
            generator_training_iteration(
                x, generator, discriminator, g_optimizer, latent_dim, latent_distribution, device
            )
        )
    return d_losses, g_losses

