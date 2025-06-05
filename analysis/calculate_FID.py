import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from load_data import load_mnist, load_config
from models.load_model import load_vae_model, load_generator_model, load_discriminator_model
from generate_dataset import generate_dataset_vae, generate_dataset_wgan, generate_data
from time import time
from scipy.stats import wasserstein_distance_nd
from scipy.spatial.distance import jensenshannon
from ignite.metrics import FID

from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

def preprocess_uint8_for_inception(images):
    # Repeat to 3 channels, FID expects three channels
    images = images.repeat(1, 3, 1, 1)
    # Resize to 299x299
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    return images

def unpack_mnist_batches(train_loader, n_images):
    all_mnist_images = []

    for batch_images, _ in train_loader:
        all_mnist_images.append(batch_images)

    mnist_images_tensor = torch.cat(all_mnist_images, dim=0)

    # Shuffle if N < 60000
    num_samples = mnist_images_tensor.size(0)
    indices = torch.randperm(num_samples)
    mnist_images_tensor = mnist_images_tensor[indices]

    mnist_images_tensor = mnist_images_tensor[:n_images, :, :, :]
    return mnist_images_tensor



FID_metrics = {
    'FID_vae': [],
    'FID_wgan_setup_1': [],
    'FID_wgan_setup_2': []
}

n_images = 1000


# Load config and MNIST data
config = load_config("configs/vae_config.yaml")
train_loader, _, _, _ = load_mnist(config)
mnist_images_tensor = unpack_mnist_batches(train_loader, n_images)

# Preprocessing of MNIST for FID
mnist_images_tensor = (mnist_images_tensor * 255).clamp(0,255).to(torch.uint8)
mnist_images_tensor = preprocess_uint8_for_inception(mnist_images_tensor)



# VAE

# Big loop: for generations i from 0 to 20
for i in range(20):
    current_dataset = generate_data(model_type='vae', label="experiment_4", gen_nr=i, n_datapoints=n_images)
    current_dataset_tensor = torch.stack(current_dataset)
    print(f"Dataset generation {i} generated")

    print(type(current_dataset_tensor), current_dataset_tensor.shape)
    print(type(mnist_images_tensor), mnist_images_tensor.shape)

    # Preprocessing (images are in [0.0, 1.0] float format), must convert to torch.uint8
    current_dataset_tensor = (current_dataset_tensor * 255).clamp(0,255).to(torch.uint8)
    current_dataset_tensor = preprocess_uint8_for_inception(current_dataset_tensor)

    fid = FrechetInceptionDistance(feature=64)
    fid.update(mnist_images_tensor, real=True)
    fid.update(current_dataset_tensor, real=False)
    fid_metric = float(fid.compute())
    print(fid_metric)
    FID_metrics['FID_vae'].append(fid_metric)

# WGAN setup 1 (experiment 4)
# Big loop: for generations i from 0 to 20
for i in range(20):
    current_dataset = generate_data(model_type='wgan', label="experiment_4", gen_nr=i, n_datapoints=n_images)
    current_dataset_tensor = torch.stack(current_dataset)
    print(f"Dataset generation {i} generated")

    print(type(current_dataset_tensor), current_dataset_tensor.shape)
    print(type(mnist_images_tensor), mnist_images_tensor.shape)

    # Preprocessing (images are in [0.0, 1.0] float format), must convert to torch.uint8
    current_dataset_tensor = (current_dataset_tensor * 255).clamp(0,255).to(torch.uint8)
    current_dataset_tensor = preprocess_uint8_for_inception(current_dataset_tensor)

    fid = FrechetInceptionDistance(feature=64)
    fid.update(mnist_images_tensor, real=True)
    fid.update(current_dataset_tensor, real=False)
    fid_metric = float(fid.compute())
    print(fid_metric)
    FID_metrics['FID_wgan_setup_1'].append(fid_metric)

# WGAN setup 2 (experiment 5)
# Big loop: for generations i from 0 to 20
for i in range(20):
    current_dataset = generate_data(model_type='wgan', label="experiment_5", gen_nr=i, n_datapoints=n_images)
    current_dataset_tensor = torch.stack(current_dataset)
    print(f"Dataset generation {i} generated")

    print(type(current_dataset_tensor), current_dataset_tensor.shape)
    print(type(mnist_images_tensor), mnist_images_tensor.shape)

    # Preprocessing (images are in [0.0, 1.0] float format), must convert to torch.uint8
    current_dataset_tensor = (current_dataset_tensor * 255).clamp(0,255).to(torch.uint8)
    current_dataset_tensor = preprocess_uint8_for_inception(current_dataset_tensor)

    fid = FrechetInceptionDistance(feature=64)
    fid.update(mnist_images_tensor, real=True)
    fid.update(current_dataset_tensor, real=False)
    fid_metric = float(fid.compute())
    print(fid_metric)
    FID_metrics['FID_wgan_setup_2'].append(fid_metric)

print(FID_metrics)
df = pd.DataFrame(FID_metrics)
df.to_csv('FID_metrics.csv', index=False)