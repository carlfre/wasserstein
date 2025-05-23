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

        
        print(f"Generation {i} complete. Loss: {loss_per_generation[i]}")
        print(f"Time taken: {time() - start_time} seconds")
        
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


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Train VAE and WGAN models.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--vae", action="store_true", help="Train VAE model.")
    group.add_argument("--wgan", action="store_true", help="Train WGAN model.")

    parser.add_argument(
        "--n_generations",
        type=int,
        default=1,
        help="Number of generations to run.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=60000,
        help="Size of the dataset to generate.",
    )
    parser.add_argument(
        "--run_label",
        type=str,
        default="",
        help="Label for the run.",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=1,
        help="Number of threads to use.",
    )
    parser.add_argument(
        "--n_interop_threads",
        type=int,
        default=1,
        help="Number of interop threads to use.",
    )

    args = parser.parse_args()

    if args.n_threads < 1 or args.n_interop_threads < 1:
        parser.error("n_threads and n_interop_threads must be >= 1")


    # Set the number of threads for PyTorch
    torch.set_num_threads(args.n_threads)
    torch.set_num_interop_threads(args.n_interop_threads)

    if args.vae:
        big_loop_vae(args.n_generations, args.dataset_size, args.run_label)
    if args.wgan:
        big_loop_wgan(args.n_generations, args.dataset_size, args.run_label)

