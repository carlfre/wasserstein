import numpy as np
import torch
import matplotlib.pyplot as plt

#TODO: write some code for plotting an image nicely

def plot_image(image: torch.Tensor | np.ndarray, filename=None, show=True) -> None:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if len(image.shape) == 3:
        if image.shape[0] != 1:
            raise ValueError("Image is 3d for some reason?")
        
        image = image[0]
    
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    


if __name__ == "__main__":
    from models.load_model import load_vae_model
    from generate_dataset import generate_dataset_vae
    from load_data import load_config

    config = load_config("configs/vae_config.yaml")
    model_placement = "checkpoints/gen_2_vae_model.pth"

    vae = load_vae_model(
        config,
        model_placement,
    )

    dataset, dataloader = generate_dataset_vae(vae, 10, config)

    # Plot an image from the dataset
    sample_index = 0  # Choose the first sample
    sample = dataset[sample_index]
    image = sample[0]  # Get the image (assuming dataset returns (image, label) pairs)
    plot_image(image, filename="sample_image.png")




# TODO: Write some code for plotting a grid of images