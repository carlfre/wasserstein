# from run import device, disc_model
from models.discriminators import DiscriminatorWGAN
import torch
from torch.autograd import grad as torch_grad


def gradient_penalty(
    real_data: torch.Tensor,
    generated_data: torch.Tensor,
    lambda_gp: float,
    discriminator: DiscriminatorWGAN,
    device: str,
) -> torch.Tensor:
    
    batch_size = real_data.size(0)

    # Calculate interpolation
    alpha = torch.rand(real_data.shape[0], 1, 1, 1, requires_grad=True, device=device)
    interpolated = alpha * real_data + (1 - alpha) * generated_data

    # Calculate probability of interpolated examp
    proba_interpolated = discriminator(interpolated)
    # Calculate gradients of probabilities
    gradients = torch_grad(
        outputs=proba_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(proba_interpolated.size(), device=device),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    return lambda_gp * ((gradients_norm - 1) ** 2).mean()
