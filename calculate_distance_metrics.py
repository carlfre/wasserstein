import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pandas as pd

from load_data import load_mnist, load_config
from models.load_model import load_vae_model, load_generator_model, load_discriminator_model
from generate_dataset import generate_dataset_vae, generate_dataset_wgan
from time import time

# VAE

wasserstein_metrics = []
KLD_metrics = []
FID_metrics = []

# Load MNIST dataset with appropriate transform (identity?) 
config = load_config("configs/vae_config.yaml")

train_loader, _, _, _ = load_mnist(config)
mnist_images = []

for images, labels in train_loader:
    mnist_images.extend(images)  # images is a batch of tensors

mnist_images = mnist_images[:10000]

mnist_images = [img.numpy() for img in mnist_images]
mnist_images_flattened = [img.flatten() for img in mnist_images]



# Big loop: for generations i from 0 to 20
for i in range(20):

    # Initialise model of generation i
    

    # Generate dataset from generation i

    # Calculate Wasserstein, KLD and FID between current dataset and MNIST

    # Save metrics




######################################################


# WGAN

wasserstein_metrics = []
KLD_metrics = []
FID_metrics = []

# Load MNIST dataset with appropriate transform (identity?) 

# Flatten MNIST samples

# Big loop: for generations i from 0 to 20

    # Initialise model of generation i

    # Generate dataset from generation i

    # Calculate Wasserstein, KLD and FID between current dataset and MNIST

    # Save metrics