import torch
import torch.nn as nn


def vae_loss(
    x: torch.Tensor, x_hat: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD
