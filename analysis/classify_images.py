import numpy as np

from collections import Counter

import torch

from models.load_model import load_cnn_model
from generate_dataset import generate_data
from plotting import plot_image

cnn = load_cnn_model()


def classify_images(images: list[torch.Tensor]) -> Counter:
    """
    Classify a list of images using the CNN model.

    Args:
        images (list[torch.Tensor]): List of images to classify.

    Returns:
        list[int]: List of predicted class indices for each image.
    """
    cnn.eval()
    with torch.no_grad():
        images = torch.stack(images).to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
    return Counter(predicted.cpu().numpy().tolist())


def classify_image(image: torch.Tensor) -> int:
    """
    Classify a single image using the CNN model.

    Args:
        image (torch.Tensor): Image to classify.

    Returns:
        int: Predicted class index for the image.
    """
    cnn.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        output = cnn(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()








# gen_nr = 10
# image = generate_data("vae", gen_nr, 1, label="experiment_4")[0][0]
# print(classify_image(image))
# plot_image(image)

from typing import Literal
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(palette="deep", font_scale=1.3)


def plot_image_classifications(model_type: Literal["vae", "wgan"], label: Literal["experiment_4", "experiment_5"]):
    n_gens = 19
    n_images = 1_000
    relative_frequency = np.zeros((10, n_gens))

    for gen_nr in range(n_gens):
        images = generate_data(model_type, gen_nr, n_images, label=label)
        predictions = classify_images(images)
        for num, count in predictions.items():
            relative_frequency[num, gen_nr] = count / n_images

    plt.plot(relative_frequency.T, alpha=0.5)
    plt.legend(
        [f"{i}" for i in range(10)], bbox_to_anchor=(1, 1), loc='upper left', 
    )
    plt.tight_layout()
    plt.xlabel("Generation")
    plt.ylabel("Relative Frequency")
    plt.savefig(
        f"plots/classification_{model_type}_{label}.pdf", bbox_inches="tight", pad_inches=0
    )
    plt.show()




# plot_image_classifications("vae", "experiment_4")
plot_image_classifications("wgan", "experiment_4")
plot_image_classifications("wgan", "experiment_5")


