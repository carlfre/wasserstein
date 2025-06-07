import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random

from load_data import load_mnist, load_config
from models.load_model import load_vae_model, load_generator_model, load_discriminator_model
from generate_dataset import generate_dataset_vae, generate_dataset_wgan, generate_data
from time import time
from scipy.stats import wasserstein_distance_nd
from scipy.spatial.distance import jensenshannon
from ignite.metrics import FID

from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

wasserstein_metrics = {
    'Wass_vae': [],
    'Wass_wgan_setup_1': [],
    'Wass_wgan_setup_2': []
}

# Load config and MNIST data
config = load_config("configs/vae_config.yaml")
n_images = 500

train_loader, _, _, _ = load_mnist(config)

mnist_images = []

for images, labels in train_loader:
    mnist_images.extend(images) # images is a batch of tensors

mnist_images = [img.numpy() for img in mnist_images]
mnist_images_flattened = [img.flatten() for img in mnist_images]

# Shuffle if N < 60000 and slice
random.shuffle(mnist_images_flattened)
mnist_images_flattened = mnist_images_flattened[:n_images]


# VAE

# Big loop: for generations i from 0 to 20
for i in range(20):
    current_dataset = generate_data(model_type='vae', label="experiment_4", gen_nr=i, n_datapoints=n_images)
    current_dataset_tensor = torch.stack(current_dataset)
    current_flattened = [img.numpy().flatten() for img in current_dataset]
    print(f"Dataset generation {i} generated")
    wass_metric = wasserstein_distance_nd(mnist_images_flattened, current_flattened)
    print(wass_metric)
    wasserstein_metrics['Wass_vae'].append(wass_metric)


# WGAN setup 1 (experiment 4)

# Big loop: for generations i from 0 to 20
for i in range(20):
    current_dataset = generate_data(model_type='wgan', label="experiment_4", gen_nr=i, n_datapoints=n_images)
    current_dataset_tensor = torch.stack(current_dataset)
    current_flattened = [img.numpy().flatten() for img in current_dataset]
    print(f"Dataset generation {i} generated")
    wass_metric = wasserstein_distance_nd(mnist_images_flattened, current_flattened)
    print(wass_metric)
    wasserstein_metrics['Wass_wgan_setup_1'].append(wass_metric)


# WGAN setup 2 (experiment 5)

# Big loop: for generations i from 0 to 20
for i in range(20):
    current_dataset = generate_data(model_type='wgan', label="experiment_5", gen_nr=i, n_datapoints=n_images)
    current_dataset_tensor = torch.stack(current_dataset)
    current_flattened = [img.numpy().flatten() for img in current_dataset]
    print(f"Dataset generation {i} generated")
    wass_metric = wasserstein_distance_nd(mnist_images_flattened, current_flattened)
    print(wass_metric)
    wasserstein_metrics['Wass_wgan_setup_2'].append(wass_metric)

print(wasserstein_metrics)
df = pd.DataFrame(wasserstein_metrics)
df.to_csv('wasserstein_metrics.csv', index=False)