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
from generate_dataset import generate_dataset_vae, generate_dataset_wgan, generate_data
from time import time
#from scipy.stats import wasserstein_distance_nd
from scipy.spatial.distance import jensenshannon
from ignite.metrics import FID

# load FID model

# from collections import OrderedDict

# import torch
# from torch import nn, optim

# from ignite.engine import *
# from ignite.handlers import *
# from ignite.metrics import *
# from ignite.metrics.clustering import *
# from ignite.metrics.regression import *
# from ignite.utils import *

# create default evaluator for doctests

# def eval_step(engine, batch):
#     return batch

# default_evaluator = Engine(eval_step)

# create default optimizer for doctests

# param_tensor = torch.zeros([1], requires_grad=True)
# default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`

# def get_default_trainer():

#     def train_step(engine, batch):
#         return batch

#     return Engine(train_step)

# create default model for doctests

# default_model = nn.Sequential(OrderedDict([
#     ('base', nn.Linear(4, 2)),
#     ('fc', nn.Linear(2, 1))
# ]))

# manual_seed(666)


########################################################

# VAE

wasserstein_metrics = []
KLD_metrics = []
FID_metrics = []

# Load MNIST dataset with appropriate transform (identity?) 
config = load_config("configs/vae_config.yaml")
n_images = 500

train_loader, _, _, _ = load_mnist(config)
mnist_images = []

for images, labels in train_loader:
    mnist_images.extend(images)  # images is a batch of tensors

mnist_images = mnist_images[:n_images]

mnist_images = [img.numpy() for img in mnist_images]
mnist_images_flattened = [img.flatten() for img in mnist_images]



# Big loop: for generations i from 0 to 20
for i in range(20):
    print(f"pre gen {i}")
    # Initialise model of generation i
    current_dataset = generate_data(model_type='vae', gen_nr=i, n_datapoints=n_images)
    print(f"post gen {i}")
    # Generate dataset from generation i

    # Calculate Wasserstein, KLD and FID between current dataset and MNIST
    current_flattened = [img.numpy().flatten() for img in current_dataset]

    # Wasserstein
    # wasserstein = wasserstein_distance_nd(mnist_images_flattened, current_flattened)
    # print(wasserstein)

    # Jense-Shannon divergence
    JSD = jensenshannon(mnist_images_flattened, current_flattened)
    print(JSD)

    # FID
    # metric = FID(num_features=1, feature_extractor=default_model)
    # metric.attach(default_evaluator, "fid")
    # state = default_evaluator.run([[torch.stack(current_dataset), torch.stack(mnist_images)]])
    # print(state.metrics["fid"])

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