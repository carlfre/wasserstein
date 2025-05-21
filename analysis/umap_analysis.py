import pandas as pd
import umap
import seaborn as sns
import torch
import matplotlib.pyplot as plt

from models.load_model import load_vae_model, load_generator_model
from generate_dataset import generate_dataset_vae, generate_dataset_wgan
from load_data import load_config
from generate_dataset import generate_data

sns.set()


model_type = "vae"
gen_nr = 0
n_datapoints = 10000

images = generate_data(model_type, gen_nr, n_datapoints)
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
images = torch.stack(images)

reducer = umap.UMAP()
embedding = reducer.fit_transform(images.view(images.size(0), -1).cpu().numpy())

sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], s=3, alpha=0.5)
plt.show()


