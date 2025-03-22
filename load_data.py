from typing import Literal

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def load_mnist(batch_size: int, transform=None):
    image_path = './data'
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

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

