from typing import Literal

import matplotlib.pyplot as plt
import torch

from models.load_model import load_vae_model, load_generator_model
from generate_dataset import generate_dataset_vae, generate_dataset_wgan
from load_data import load_config


import torch
import numpy as np
from torchvision import models

# Load pretrained ResNet50 and remove classification layer
resnet = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = feature_extractor.to(device)

def convert_grayscale_to_rgb(img):
    """Convert a grayscale tensor [1, H, W] to 3-channel RGB [3, H, W]."""
    return img.repeat(3, 1, 1)

def extract_features_from_grayscale_tensor_list(image_tensors):
    """
    Takes a list of grayscale image tensors [1 x H x W] and returns feature matrix.
    """
    features = []
    with torch.no_grad():
        for img in image_tensors:
            img_rgb = convert_grayscale_to_rgb(img)
            img_rgb = img_rgb.unsqueeze(0).to(device)  # Add batch dimension
            feat = feature_extractor(img_rgb).squeeze().cpu().numpy().flatten()
            features.append(feat)
    return np.vstack(features)  # shape: [num_images, feature_dim]



model_type: Literal["vae", "wgan"] = "wgan"
gen_nr = 0
n_datapoints = 1000

if  model_type == "vae":
    config = load_config("configs/vae_config.yaml")
    vae_path = f"checkpoints/gen_{gen_nr}_vae_experiment_4.pth"
    vae = load_vae_model(config, vae_path)
    dataset, _ = generate_dataset_vae(vae, n_datapoints, config)
elif model_type == "wgan":
    config = load_config("configs/wgan_config.yaml")
    generator_path = f"checkpoints/gen_{gen_nr}_wgan_generator_experiment_4.pth"
    generator = load_generator_model(config, generator_path)
    dataset, _ = generate_dataset_wgan(generator, n_datapoints, config)
else:
    raise ValueError("Invalid model type. Choose 'vae' or 'wgan'.")

images = [sample[0] for sample in dataset]



from torchvision import transforms
from PIL import Image

# Preprocessing for grayscale
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(),  # Ensures output is 1-channel
    transforms.ToTensor(),
])

# Load and preprocess grayscale images
# image_tensors = [transform(Image.open(p)) for p in img_paths]

# images = [transform(img) for img in images]  # Assuming images are PIL Images

# Extract features
features = extract_features_from_grayscale_tensor_list(images)
print(features.shape)  # Should be [num_images, feature_dim]

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# tsne = TSNE(n_components=2, perplexity=30)
# X_2d = tsne.fit_transform(features)

# plt.scatter(X_2d[:, 0], X_2d[:, 1])
# plt.title("Image Feature Clusters")
# plt.show()


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
k_range = range(1, 25)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertias.append(kmeans.inertia_)

plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
