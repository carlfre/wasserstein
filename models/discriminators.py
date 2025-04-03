import torch.nn as nn

class DiscriminatorWGAN(nn.Module):
    def __init__(self, n_channel_scaling_factor):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_channel_scaling_factor, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_channel_scaling_factor, n_channel_scaling_factor*2, 4, 2, 1,
            bias=False),
            nn.InstanceNorm2d(n_channel_scaling_factor * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_channel_scaling_factor*2, n_channel_scaling_factor*4, 3, 2, 1,
            bias=False),
            nn.InstanceNorm2d(n_channel_scaling_factor*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_channel_scaling_factor*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(0)




