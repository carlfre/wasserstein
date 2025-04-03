import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import yaml

from models.discriminators import DiscriminatorWGAN
from models.generators import make_generator_network_wgan
from models.discriminators import DiscriminatorWGAN
from training.train_wgan import generator_training_iteration, discriminator_training_iteration, wgan_train_epoch
from utils import create_noise, create_samples
from load_data import load_mnist




with open("configs/wgan_config.yaml") as f:
    config = yaml.safe_load(f)

    training_config = config["training"]
    model_config = config["model_specifics"]


    batch_size = training_config["batch_size"]
    transform = training_config["transform"]
    device = training_config["device"]
    n_epochs = training_config["n_epochs"]
    learning_rate = training_config["learning_rate"]

    n_channel_scaling_factor = model_config["n_channel_scaling_factor"]
    lambda_gp = model_config["lambda_gp"]
    mode_z = model_config["mode_z"]
    z_size = model_config["z_size"]
    n_critic_iterations = model_config["n_critic_iterations"]


    train_loader, test_loader, train_set, test_set = load_mnist(
        batch_size, transform=transform
    )
    print("Data loaded.")

    fixed_z = create_noise(batch_size, z_size, mode_z).to(device)


    generator = make_generator_network_wgan(z_size, n_channel_scaling_factor).to(device)
    discriminator = DiscriminatorWGAN(n_channel_scaling_factor).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), learning_rate)


    print("Models generated.")


    print("starting training.")
    epoch_samples_wgan = []
    torch.manual_seed(1)
    for epoch in range(1, n_epochs + 1):
        d_losses, g_losses = wgan_train_epoch(
            (generator, discriminator),
            (g_optimizer, d_optimizer),
            train_loader,
            config,
        )

        print(f"Epoch {epoch:03d} | D Loss >>" f" {torch.FloatTensor(d_losses).mean():.4f}")
        generator.eval()
        epoch_samples_wgan.append(
            create_samples(generator, fixed_z, batch_size, (1, 28, 28))
            .detach()
            .cpu()
            .numpy(),
        )
        np.save(
            f"/home/carlfre/uni/wasserstein/wasserstein_rashka/output/generated_{epoch}.npy",
            epoch_samples_wgan[-1],
        )
        # Save the weights of generator and discriminator as checkpoints
        torch.save(
            generator.state_dict(),
            f"/home/carlfre/uni/wasserstein/wasserstein_rashka/checkpoints/gen_model_epoch_{epoch}.pth",
        )
        torch.save(
            discriminator.state_dict(),
            f"/home/carlfre/uni/wasserstein/wasserstein_rashka/checkpoints/disc_model_epoch_{epoch}.pth",
        )


