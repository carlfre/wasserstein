import torch
import torch.nn as nn


class GeneratorV1(nn.Module):
    r""" An Generator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1704.00028v3>`_ paper.
    """

    def __init__(self, size_of_noise_vector: int = 100, n_filters: int = 32):
        # self.super().__init__()
        super().__init__()
        # super(GeneratorV1, self).__init__() # maybe replace with above line?

        self.main = nn.Sequential(
            nn.ConvTranspose2d(size_of_noise_vector, n_filters*4, 4,
            1, 0, bias=False),
            nn.BatchNorm2d(n_filters*4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_filters*4, n_filters*2,
            3, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_filters*2, n_filters,
            4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_filters, 1, 4, 2, 1,
            bias=False),
            nn.Tanh()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Defines the computation performed at every call.

        Args:
            input (tensor): input tensor into the calculation.

        Returns:
            A four-dimensional vector (NCHW).
        """
        out = self.main(input)
        return out


