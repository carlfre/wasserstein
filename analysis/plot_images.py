import matplotlib.pyplot as plt

from generate_dataset import generate_data
from plotting import plot_grid


def main():
    n_images = 20
    plotting_generations = [0, 4, 9, 19] #1-indexed, these are generations 1, 5, 10, 20.

    for gen_nr in plotting_generations:
        images = generate_data("wgan", gen_nr, n_images, label="experiment_4")
        plot_grid(images)
        plt.savefig(f"plots/generated_images_wgan_gen_{gen_nr}_exp4.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()

        images = generate_data("wgan", gen_nr, n_images, label="experiment_5")
        plot_grid(images)
        plt.savefig(f"plots/generated_images_wgan_gen_{gen_nr}_exp5.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()

        images = generate_data("vae", gen_nr, n_images, label="experiment_4")
        plot_grid(images)
        plt.savefig(f"plots/generated_images_vae_gen_{gen_nr}.pdf", bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == "__main__":
    main()
