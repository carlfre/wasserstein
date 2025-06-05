import torch
import torch.nn.functional as F
import torch.optim as optim

from load_data import load_mnist, load_config
from models.cnn import CNN_MNIST


config = load_config("configs/cnn_config.yaml")
training_config = config["training"]
device = training_config["device"]
n_epochs = training_config["n_epochs"]
learning_rate = training_config["learning_rate"]
train_loader, test_loader, train_set, test_set = load_mnist(config)


def train_cnn(model, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}]  Loss: {loss.item():.4f}"
            )


def test_cnn(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


def main():
    model = CNN_MNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        train_cnn(model, device, train_loader, optimizer, epoch)
        test_cnn(model, device, test_loader)


    torch.save(model.state_dict(), f"checkpoints/cnn.pth")


# You can run the file as a script to train the model.
if __name__ == "__main__":
    main()
