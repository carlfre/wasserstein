import torch
from models.discriminators import DiscriminatorWGAN
from models.generators import make_generator_network_wgan
from models.vae import VAE, Decoder, Encoder



def load_vae_model(vae_path: str, config: dict[str, dict]):
    """
    Load the VAE model from the specified path.
    """
    training_config = config["training"]
    model_config = config["model_specifics"]

    device = training_config["device"]
    
    image_dim = list(map(int, model_config["image_dim"].split()))
    hidden_dim = model_config["hidden_dim"]
    latent_dim = model_config["latent_dim"]

    flattened_dim = image_dim[0] * image_dim[1]

    encoder = Encoder(flattened_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, flattened_dim)
    vae = VAE(encoder, decoder, device)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.to(device)
    vae.eval()
    return vae


def load_discriminator_model(discriminator_path: str, config: dict[str, dict]):
    """
    Load the discriminator model from the specified path.
    """
    training_config = config["training"]
    model_config = config["model_specifics"]

    device = training_config["device"]

    n_channel_scaling_factor = model_config["n_channel_scaling_factor"]
    
    discriminator = DiscriminatorWGAN(n_channel_scaling_factor)
    discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
    discriminator.to(device)
    discriminator.eval()
    return discriminator


def load_generator_model(generator_path: str, config: dict[str, dict]):
    """
    Load the generator model from the specified path.
    """
    training_config = config["training"]
    model_config = config["model_specifics"]

    device = training_config["device"]

    n_channel_scaling_factor = model_config["n_channel_scaling_factor"]
    z_size = model_config["z_size"]

    generator = make_generator_network_wgan(z_size, n_channel_scaling_factor)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.to(device)
    generator.eval()
    return generator




if __name__ == "__main__":
    from load_data import load_config

    config = load_config("configs/wgan_config.yaml")

    discriminator = load_discriminator_model(
        "checkpoints/disc_model_epoch_26.pth",
        config
    )

    print("sld")
    generator = load_generator_model(
        "checkpoints/gen_model_epoch_26.pth",
        config
    )

    print("jdljl")
    config = load_config("configs/vae_config.yaml")
    vae = load_vae_model(
        "checkpoints/vae_model_epoch_3.pth",
        config
    )


    from torchvision.utils import save_image, make_grid



    training_config = config["training"]
    model_config = config["model_specifics"]


    batch_size = training_config["batch_size"]

    transform = training_config["transform"]
    device = training_config["device"]
    n_epochs = training_config["n_epochs"]
    learning_rate = training_config["learning_rate"]

    hidden_dim = model_config["hidden_dim"]
    latent_dim = model_config["latent_dim"]
    constant_noise = torch.randn(batch_size, latent_dim).to(device)
    generated_images = vae.decoder(constant_noise)
    save_image(generated_images.view(batch_size, 1, 28, 28), f"output/does_it_load.png")



