import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.vae import VAE


from loss_functions.vae_loss import vae_loss

def vae_train_epoch(
        model: VAE,
        optimizer: Optimizer,
        data_loader: DataLoader,
        config: dict,
) -> float:
    training_config = config["training"]

    batch_size = training_config["batch_size"]
    device = training_config["device"]

    # # Get x_dim, i.e. the size of the image. Should be constant across batches
    # data_iter = iter(data_loader)
    # x, _ = next(data_iter)
    # x_dim = x.size(2) * x.size(3)

    losses = []
    for batch_idx, (x, _) in enumerate(data_loader):
        # print(x.size(0), x.size(1), x.size(2), x.size(3))
        x_dim = x.size(2) * x.size(3)
        x = x.view(-1, x_dim)
        x = x.to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = vae_loss(x, x_hat, mean, log_var)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
    return losses