

# %%
from tqdm import tqdm
import matplotlib.pyplot as plt

with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, x_dim)
        x = x.to(device)

        x_hat, _, _ = model(x)

        break


# %%
def show_image(x, idx):
    x = x.view(batch_size, 28, 28)

    fig = plt.figure()
    plt.imshow(x[idx].cpu().numpy())


show_image(x, 1)


# %%
show_image(x_hat, 1)

# %%


with torch.no_grad():
    constant_noise = torch.randn(batch_size, latent_dim).to(device)
    generated_images = decoder(constant_noise)

save_image(generated_images.view(batch_size, 1, 28, 28), "generated_sample.png")


# %%
show_image(generated_images, idx=27)
