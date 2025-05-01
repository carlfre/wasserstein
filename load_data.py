from typing import Literal
import yaml

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def load_mnist(config: dict[str]):

    training_config = config["training"]
    batch_size = training_config["batch_size"]
    transform = training_config["transform"]

    image_path = './data'


    if transform == "normalize":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    elif transform == "identity":
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
    else:
        raise ValueError("Invalid transform argument. Use 'identity' or 'normalize'.")


    train_set = torchvision.datasets.MNIST(
        root=image_path, train=True,
        transform=transform, download=True
    )
    test_set = torchvision.datasets.MNIST(
        root=image_path, train=False,
        transform=transform, download=True
    )

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_set,  batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_set, test_set


def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config
    

