import torch
import numpy as np
from torch.autograd import grad as torch_grad
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from models.generators import make_generator_network_wgan
from models.discriminators import DiscriminatorWGAN
from utils import create_noise, create_samples
from load_data import load_mnist



z_size = 100
n_filters = 64
device = 'cuda'

lambda_gp = 10
batch_size = 64
mode_z = 'uniform'

mnist_dataset = load_mnist()
mnist_dl = DataLoader(
    mnist_dataset, batch_size=batch_size,
    shuffle=True, drop_last=True
)

fixed_z = create_noise(batch_size, z_size, mode_z).to(device)


print("Data loaded.")


gen_model = make_generator_network_wgan(
    z_size, n_filters
).to(device)
disc_model = DiscriminatorWGAN(n_filters).to(device)
g_optimizer = torch.optim.Adam(gen_model.parameters(), 0.0002)
d_optimizer = torch.optim.Adam(disc_model.parameters(), 0.0002)


print("Models generated.")

def gradient_penalty(real_data, generated_data):
    batch_size = real_data.size(0)

    # Calculate interpolation
    alpha = torch.rand(real_data.shape[0], 1, 1, 1,
    requires_grad=True, device=device)
    interpolated = alpha * real_data + \
    (1 - alpha) * generated_data

    # Calculate probability of interpolated examp
    proba_interpolated = disc_model(interpolated)
    # Calculate gradients of probabilities
    gradients = torch_grad(
    outputs=proba_interpolated, inputs=interpolated,
    grad_outputs=torch.ones(proba_interpolated.size(),
    device=device),
    create_graph=True, retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)
    return lambda_gp * ((gradients_norm - 1)**2).mean()


def d_train_wgan(x):
    disc_model.zero_grad()

    batch_size = x.size(0)
    x = x.to(device)

    # Calculate probabilities on real and generated data
    d_real = disc_model(x)
    input_z = create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z)
    d_generated = disc_model(g_output)
    d_loss = d_generated.mean() - d_real.mean() + \
    gradient_penalty(x.data, g_output.data)
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item()

def g_train_wgan(x):
    gen_model.zero_grad()

    batch_size = x.size(0)
    input_z = create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z)

    d_generated = disc_model(g_output)
    g_loss = -d_generated.mean()

    # gradient backprop & optimize ONLY G's parameters
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()



print("starting training.")
epoch_samples_wgan = []
lambda_gp = 10.0
num_epochs = 100
torch.manual_seed(1)
critic_iterations = 5
for epoch in range(1, num_epochs+1):
    gen_model.train()
    d_losses, g_losses = [], []
    for i, (x, _) in enumerate(mnist_dl):
        for _ in range(critic_iterations):
            d_loss = d_train_wgan(x)
        d_losses.append(d_loss)
        g_losses.append(g_train_wgan(x))
    # print("Epoch number: ", epoch, "dloss:", d_losses[-1], "gloss:", g_losses[-1], end='\r')


    print(f'Epoch {epoch:03d} | D Loss >>'
    f' {torch.FloatTensor(d_losses).mean():.4f}')
    gen_model.eval()
    epoch_samples_wgan.append(
        create_samples(
                gen_model, fixed_z, batch_size, (1, 28, 28)
            ).detach().cpu().numpy(),
    )
    np.save(f'/home/carlfre/uni/wasserstein/wasserstein_rashka/output/generated_{epoch}.npy', epoch_samples_wgan[-1])
    # Save the weights of generator and discriminator as checkpoints
    torch.save(gen_model.state_dict(), f'/home/carlfre/uni/wasserstein/wasserstein_rashka/checkpoints/gen_model_epoch_{epoch}.pth')
    torch.save(disc_model.state_dict(), f'/home/carlfre/uni/wasserstein/wasserstein_rashka/checkpoints/disc_model_epoch_{epoch}.pth')

# np.save('/home/carlfre/uni/wasserstein/wasserstein_rashka/generated_samples.npy', epoch_samples_wgan[-1])


# Save the generated samples as a tensor
# samples.save('/home/carlfre/uni/wasserstein/wasserstein_rashka/generated_samples.pt')
# torch.save(torch.tensor(epoch_samples_wgan), '/home/carlfre/uni/wasserstein/wasserstein_rashka/generated_samples.pt')
# print("Generated samples saved.")
