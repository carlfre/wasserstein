import yaml
from torchvision.utils import save_image, make_grid
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd

from load_data import load_mnist, load_config
from loss_functions.vae_loss import vae_loss
from models.load_model import load_vae_model, load_generator_model, load_discriminator_model
from training.train_vae import train_vae
from training.train_wgan import train_wgan
from generate_dataset import generate_dataset_vae, generate_dataset_wgan
from time import time
# from utils import 

def big_loop_vae(n_generations: int, dataset_size: int, run_label: str = "") -> None:
    config = load_config("configs/vae_config.yaml")

    train_loader, test_loader, train_set, test_set = load_mnist(config)
    previous_dataloader = train_loader
    
    loss_per_generation: dict[int, float] = {}
    for i in range(n_generations):
        start_time = time()

        model = load_vae_model(config)
        model, generation_losses = train_vae(model, previous_dataloader, config)
        # loss_per_generation.append(generation_losses[-1])
        loss_per_generation[i] = generation_losses[-1]

        if i == 1:
            dataiter = iter(previous_dataloader)
            sample = next(dataiter)
            print(f"Sample shape: {sample.shape}")
        print(f"Generation {i} complete. Loss: {loss_per_generation[i]}")
        print(f"Time taken: {time() - start_time} seconds")
        print(len(previous_dataloader))
        print(previous_dataloader.dataset.data.shape)
        
        torch.save(
            model.state_dict(),
            f"checkpoints/gen_{i}_vae_{run_label}.pth"
        )
        _, previous_dataloader = generate_dataset_vae(model, dataset_size, config)

    df = pd.DataFrame.from_dict(loss_per_generation, orient="index", columns=["loss"])
    df.to_csv(f'output/losses/losses_vae_{run_label}.csv', index_label="generation")

    print("Done!")


def big_loop_wgan(n_generations: int, dataset_size: int, run_label: str = "" ) -> None:
    config = load_config("configs/wgan_config.yaml")

    train_loader, test_loader, train_set, test_set = load_mnist(config)
    previous_dataloader = train_loader

    generator_loss_per_generation: dict[int, float] = {}
    discriminator_loss_per_generation: dict[int, float] = {}
    for i in range(n_generations):
        generator = load_generator_model(config)
        discriminator = load_discriminator_model(config)

        generator, discriminator, g_losses, d_losses = train_wgan(
            generator,
            discriminator,
            previous_dataloader,
            config
        )
        generator_loss_per_generation[i] = g_losses[-1]
        discriminator_loss_per_generation[i] = d_losses[-1]
        print(f"Generation {i} complete.")
        print(f"Generator loss: {generator_loss_per_generation[i]}. Discriminator loss: {discriminator_loss_per_generation[i]}")
        torch.save(
            generator.state_dict(),
            f"checkpoints/gen_{i}_wgan_generator_{run_label}.pth"
        )
        torch.save(
            discriminator.state_dict(),
            f"checkpoints/gen_{i}_wgan_discriminator_{run_label}.pth"
        )
        _, previous_dataloader = generate_dataset_wgan(generator, dataset_size, config)

    # df = pd.DataFrame.from_dict(generator_loss_per_generation, orient="index", columns=["generator_loss"])
    df = pd.DataFrame({"generator_loss": generator_loss_per_generation, "discriminator_loss": discriminator_loss_per_generation})
    df.to_csv(f'output/losses/losses_wgan_{run_label}.csv', index_label="generation")
    print("Done!")

# TODO: add a big loop for wgan

if __name__ == "__main__":
    big_loop_vae(4, 60000, "debugging")
    # big_loop_wgan(30, 60000, "experiment_1")


# train_loader, test_loader, train_set, test_set = load_mnist(batch_size=batch_size)

# BCE_loss = nn.BCELoss()

# encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
# decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

# model = VAE(encoder, decoder, device).to(device)

# optimizer = Adam(model.parameters(), lr=learning_rate)

# constant_noise = torch.randn(batch_size, latent_dim).to(device)





# print("Start training VAE...")
# model.train()

# for epoch in range(n_epochs):
#     losses = vae_train_epoch(model, optimizer, train_loader, config)
#     print(
#         "\tEpoch",
#         epoch + 1,
#         "complete!",
#         "\tAverage Loss: ",
#         str(sum(losses)/len(losses)),
#     )
#     with torch.no_grad():
#         generated_images = decoder(constant_noise)
#         save_image(generated_images.view(batch_size, 1, 28, 28), f"output/vae_generated_{epoch}.png")
    
#     torch.save(
#         model.state_dict(),
#         f"checkpoints/vae_model_epoch_{epoch}.pth"
#     )

# print("Finish!!")
