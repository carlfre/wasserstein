import yaml
from torchvision.utils import save_image, make_grid
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import Adam

from load_data import load_mnist, load_config
from loss_functions.vae_loss import vae_loss
from models.load_model import load_vae_model
from training.train_vae import train_vae
from generate_dataset import generate_dataset_vae



def big_loop_vae(n_generations: int, dataset_size: int) -> None:
    config = load_config("configs/vae_config.yaml")


    # TODO: transform is hardcoded here. Maybe change?
    train_loader, test_loader, train_set, test_set = load_mnist(config)
    previous_dataloader = train_loader
    
    loss_per_generation: list[float] = []
    for i in range(n_generations):

        # model = initialize_model()
        model = load_vae_model(config)
        model, generation_losses = train_vae(model, previous_dataloader, config)
        loss_per_generation.append(generation_losses[-1])
        # TODO: Save the model
        torch.save(
            model.state_dict(),
            f"checkpoints/gen_{i}_vae_model.pth"
        )
        _, previous_dataloader = generate_dataset_vae(model, dataset_size, config)
        print(f"Generation {i} complete. Loss: {generation_losses[-1]}")

    print("Done!")

if __name__ == "__main__":
    big_loop_vae(3, 10000)


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