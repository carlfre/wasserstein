from generate_dataset import generate_dataset_vae, generate_dataset_wgan
from models.load_model import load_vae_model, load_generator_model, load_discriminator_model
from models.vae import VAE
from load_data import load_config
import os
import torch



################### WGAN

config: dict = load_config("configs/wgan_config.yaml")
generator = load_generator_model("checkpoints/gen_model_epoch_26.pth", config)
# discriminator = load_discriminator_model("checkpoints/disc_model_epoch_26.pth", config)


dataset, dataloader = generate_dataset_wgan(generator, 100, config)

# Create directory if it doesn't exist
data_dir = "data/testtest_wgan"
os.makedirs(data_dir, exist_ok=True)

# Save the dataset
torch.save(dataset, os.path.join(data_dir, "dataset.pt"))
print(f"Dataset saved to {os.path.join(data_dir, 'dataset.pt')}")



# Load the saved dataset
loaded_data_dir = "data/testtest_wgan"
loaded_dataset = torch.load(os.path.join(loaded_data_dir, "dataset.pt"))
print(f"Dataset loaded from {os.path.join(loaded_data_dir, 'dataset.pt')}")
print(f"Loaded dataset size: {len(loaded_dataset)}")


# Get an element from the dataset
sample_index = 0  # You can change this index to access different samples
sample = loaded_dataset[sample_index]

# If the dataset returns a tuple (common for image datasets with labels)
if isinstance(sample, tuple):
    image, label = sample
    print(f"Sample at index {sample_index}:")
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
else:
    # If the dataset returns just the data
    print(f"Sample at index {sample_index}:")
    print(f"Sample shape: {sample.shape}")

# # Optionally, if you want to visualize the image (if it's an image dataset)
# if hasattr(loaded_dataset, 'plot') and callable(getattr(loaded_dataset, 'plot')):
#     loaded_dataset.plot(sample_index)
import matplotlib.pyplot as plt
import numpy as np
from plot import plot_image


plot_image(sample)
# sample = sample.numpy()[0]
# plt.imshow(sample)
# plt.show()



# ############### VAE 

# config: dict = load_config("configs/vae_config.yaml")
# vae: VAE = load_vae_model("checkpoints/vae_model_epoch_3.pth", config)

# dataset, dataloader = generate_dataset_vae(vae, 100, config)
# print("Dataset generated with VAE.")

# # Create directory if it doesn't exist
# data_dir = "data/testtest"
# os.makedirs(data_dir, exist_ok=True)

# # Save the dataset
# torch.save(dataset, os.path.join(data_dir, "dataset.pt"))
# print(f"Dataset saved to {os.path.join(data_dir, 'dataset.pt')}")



# # Load the saved dataset
# loaded_data_dir = "data/testtest"
# loaded_dataset = torch.load(os.path.join(loaded_data_dir, "dataset.pt"))
# print(f"Dataset loaded from {os.path.join(loaded_data_dir, 'dataset.pt')}")
# print(f"Loaded dataset size: {len(loaded_dataset)}")


# # Get an element from the dataset
# sample_index = 0  # You can change this index to access different samples
# sample = loaded_dataset[sample_index]

# # If the dataset returns a tuple (common for image datasets with labels)
# if isinstance(sample, tuple):
#     image, label = sample
#     print(f"Sample at index {sample_index}:")
#     print(f"Image shape: {image.shape}")
#     print(f"Label: {label}")
# else:
#     # If the dataset returns just the data
#     print(f"Sample at index {sample_index}:")
#     print(f"Sample shape: {sample.shape}")

# # # Optionally, if you want to visualize the image (if it's an image dataset)
# # if hasattr(loaded_dataset, 'plot') and callable(getattr(loaded_dataset, 'plot')):
# #     loaded_dataset.plot(sample_index)
# import matplotlib.pyplot as plt
# import numpy as np

# sample = sample.numpy()[0]
# plt.imshow(sample)
# plt.show()