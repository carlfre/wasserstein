from typing import Literal

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

from generate_dataset import generate_data

def tsne_analysis(model_type: Literal["vae", "wgan"], gen_nr: int, n_datapoints: int = 10_000) -> None:
    # if  model_type == "vae":
    #     config = load_config("configs/vae_config.yaml")
    #     vae_path = f"checkpoints/gen_{gen_nr}_vae_experiment_4.pth"
    #     vae = load_vae_model(config, vae_path)
    #     dataset, _ = generate_dataset_vae(vae, n_datapoints, config)
    # elif model_type == "wgan":
    #     config = load_config("configs/wgan_config.yaml")
    #     generator_path = f"checkpoints/gen_{gen_nr}_wgan_generator_experiment_4.pth"
    #     generator = load_generator_model(config, generator_path)
    #     dataset, _ = generate_dataset_wgan(generator, n_datapoints, config)
    # else:
    #     raise ValueError("Invalid model type. Choose 'vae' or 'wgan'.")

    # images = [sample[0] for sample in dataset]
    images = generate_data(model_type, gen_nr, n_datapoints)
    images = torch.stack(images)
    images = images.view(images.size(0), -1)  # Flatten the images
    images = images.cpu().numpy()  # Convert to numpy array
    print(images.shape)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(images)
    print(tsne_results.shape)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5)
    plt.title("t-SNE Visualization of Generated Images")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid()
    plt.show()
    # plt.savefig(f"plots/tsne_vae_gen_{gen_nr}.pdf", bbox_inches='tight', pad_inches=0)
    # plt.close()


from torchvision import models, transforms
from PIL import Image
import torch
import os
import numpy as np

# Load pretrained model (e.g., ResNet without final classification layer)
resnet = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def extract_feature(img_path):
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img_t).squeeze().numpy()
    return features.flatten()

def main():
    pass


if __name__ == "__main__":
    main()