from typing import Literal

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def load_mnist(batch_size: int, transform: Literal["identity", "normalize"] = "identity"):
    image_path = './data'

    match transform:
        case "normalize":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
            ])
        case "identity":
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        case _:
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

