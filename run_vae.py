#%%

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.optim import Adam

from load_data import load_mnist
from loss_functions.vae_loss import loss_function
from models.vae import VAE, Encoder, Decoder


cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")


batch_size = 100
lr = 1e-3
epochs = 30

x_dim  = 784
hidden_dim = 400
latent_dim = 200


train_loader, test_loader, train_set, test_set = load_mnist(batch_size=batch_size)

BCE_loss = nn.BCELoss()

encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

model = VAE(encoder, decoder, DEVICE).to(DEVICE)

optimizer = Adam(model.parameters(), lr=lr)



print("Start training VAE...")
model.train()

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    
print("Finish!!")



#%%
from tqdm import tqdm
import matplotlib.pyplot as plt

with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)
        
        x_hat, _, _ = model(x)


        break

#%%
def show_image(x, idx):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())


show_image(x, 1)


#%% 
show_image(x_hat, 1)

#%%
from torchvision.utils import save_image, make_grid


with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    generated_images = decoder(noise)

save_image(generated_images.view(batch_size, 1, 28, 28), 'generated_sample.png')


#%% 
show_image(generated_images, idx=27)
