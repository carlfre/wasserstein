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
    # # Convert to float and normalize to [0, 1]
    # images = images_uint8.to(torch.float32) / 255.0
    # Repeat to 3 channels
    images = images.repeat(1, 3, 1, 1)
    # Resize to 299x299
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    return images


##########################################################################################

# VAE

wasserstein_metrics = []
#KLD_metrics = []
FID_metrics = []

# Load MNIST dataset with appropriate transform (identity?) 
config = load_config("configs/vae_config.yaml")
n_images = 500

train_loader, _, _, _ = load_mnist(config)

all_mnist_images = []

for batch_images, _ in train_loader:
    all_mnist_images.append(batch_images)

mnist_images_tensor = torch.cat(all_mnist_images, dim=0)

# Shuffle
num_samples = mnist_images_tensor.size(0)
indices = torch.randperm(num_samples)
mnist_images_tensor = mnist_images_tensor[indices]

mnist_images_tensor = mnist_images_tensor[:n_images, :, :, :]


# mnist_images = []

# for images, labels in train_loader:
#     mnist_images.extend(images)  # images is a batch of tensors

# print(type(mnist_images))
# mnist_images = mnist_images[:n_images]
# print(type(mnist_images))
# #print(mnist_images.shape)

# mnist_images = [img.numpy() for img in mnist_images]
# mnist_images_flattened = [img.flatten() for img in mnist_images]

# Preprocessing of MNIST for FID
mnist_images_tensor = (mnist_images_tensor * 255).clamp(0,255).to(torch.uint8)
mnist_images_tensor = preprocess_uint8_for_inception(mnist_images_tensor)

# Big loop: for generations i from 0 to 20
for i in range(20):
    print(f"pre gen {i}")
    # Initialise model of generation i
    current_dataset = generate_data(model_type='wgan', label="experiment_4", gen_nr=i, n_datapoints=n_images)
    current_dataset_tensor = torch.stack(current_dataset)
    print(f"post gen {i}")
    # Generate dataset from generation i

    # Calculate Wasserstein, KLD and FID between current dataset and MNIST

    # Wasserstein
    current_flattened = [img.numpy().flatten() for img in current_dataset]
    # wasserstein = wasserstein_distance_nd(mnist_images_flattened, current_flattened)
    # print(wasserstein)

    # Jense-Shannon divergence
    # JSD = jensenshannon(mnist_images_flattened, current_flattened, axis=1)
    # print(np.mean(JSD)**2)
    # print(len(JSD))

    print(type(current_dataset_tensor), current_dataset_tensor.shape)
    print(type(mnist_images_tensor), mnist_images_tensor.shape)

    # FID

        # Preprocessing (images are in [0.0, 1.0] float format), must convert to torch.uint8
    current_dataset_tensor = (current_dataset_tensor * 255).clamp(0,255).to(torch.uint8)
    current_dataset_tensor = preprocess_uint8_for_inception(current_dataset_tensor)

    fid = FrechetInceptionDistance(feature=64, input_img_size=(1,28,28))
    # fid = FrechetInceptionDistance(feature=64)
    fid.update(current_dataset_tensor, real=False)
    fid.update(mnist_images_tensor, real=True)
    fid_metric = fid.compute()
    print(fid_metric)



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