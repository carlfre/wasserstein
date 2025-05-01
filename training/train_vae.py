import torch
import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader


from models.vae import VAE, Encoder, Decoder


from loss_functions.vae_loss import vae_loss

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

def vae_train_epoch(
        model: VAE,
        optimizer: Optimizer,
        data_loader: DataLoader,
        config: dict,
) -> list[float]:
    training_config = config["training"]
    device = training_config["device"]

    losses = []
    for batch_idx, (x, _) in enumerate(data_loader):
        x = x.to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = vae_loss(x, x_hat, mean, log_var)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
    return losses


def train_vae(model: VAE, dataloader: DataLoader, config: dict[str]) -> tuple[VAE, list[float]]:

        training_config = config["training"]
        n_epochs = training_config["n_epochs"]
        learning_rate = training_config["learning_rate"]

        optimizer = Adam(model.parameters(), lr=learning_rate)
        model.train()
        losses = []
        for epoch in range(n_epochs):
            epoch_losses = vae_train_epoch(model, optimizer, dataloader, config)
            losses.append(np.mean(epoch_losses))

        return model, losses
