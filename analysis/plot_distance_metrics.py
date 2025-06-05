import matplotlib.pyplot as plt
import pandas as pd

# FID (1000 images)
FID_df = pd.read_csv("FID_metrics.csv")
FID_dict = FID_df.to_dict(orient='list')

for model in FID_dict.keys():
    if 'vae' in model:
        label = 'VAE'
    elif 'setup_1' in model:
        label = "WGAN Setup 1"
    elif 'setup_2' in model:
        label = "WGAN Setup 2"

    gens = [i+1 for i in range(20)]
    plt.plot(gens, FID_dict[model], label=label)

plt.title("Fréchet Inception Distances with 1000 images")
plt.legend()
plt.savefig("plots/FID_1000.png")
plt.show()


# Wasserstein (500 images)
wass_df = pd.read_csv("wasserstein_metrics.csv")
wass_dict = wass_df.to_dict(orient='list')

for model in wass_dict.keys():
    if 'vae' in model:
        label = 'VAE'
    elif 'setup_1' in model:
        label = "WGAN Setup 1"
    elif 'setup_2' in model:
        label = "WGAN Setup 2"

    gens = [i+1 for i in range(20)]
    plt.plot(gens, wass_dict[model], label=label)

plt.title("Wasserstein-1 distances with 500 images")
plt.legend()
plt.savefig("plots/wasserstein_500.png")
plt.show()