import yaml
from torchvision.utils import save_image, make_grid
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import Adam

from load_data import load_mnist
from loss_functions.vae_loss import vae_loss
from models.vae import VAE, Encoder, Decoder
from training.train_vae import vae_train_epoch


with open("configs/vae_config.yaml") as f:
    config = yaml.safe_load(f)

    training_config = config["training"]
    model_config = config["model_specifics"]


    batch_size = training_config["batch_size"]

    transform = training_config["transform"]
    device = training_config["device"]
    n_epochs = training_config["n_epochs"]
    learning_rate = training_config["learning_rate"]

    hidden_dim = model_config["hidden_dim"]
    latent_dim = model_config["latent_dim"]

    x_dim = 28 * 28 # Image size

    train_loader, test_loader, train_set, test_set = load_mnist(config)

    BCE_loss = nn.BCELoss()

    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

    model = VAE(encoder, decoder, device).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    constant_noise = torch.randn(batch_size, latent_dim).to(device)


    print("Start training VAE...")
    model.train()

    for epoch in range(n_epochs):
        losses = vae_train_epoch(model, optimizer, train_loader, config)
        print(
            "\tEpoch",
            epoch + 1,
            "complete!",
            "\tAverage Loss: ",
            str(sum(losses)/len(losses)),
        )
        with torch.no_grad():
            generated_images = decoder(constant_noise)
            save_image(generated_images.view(batch_size, 1, 28, 28), f"output/vae_generated_{epoch}.png")
        
        torch.save(
            model.state_dict(),
            f"checkpoints/vae_model_epoch_{epoch}.pth"
        )

    print("Finish!!")
