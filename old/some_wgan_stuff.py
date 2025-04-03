        # generator.train()
        # d_losses, g_losses = [], []
        # for i, (x, _) in enumerate(train_loader):
        #     for _ in range(n_critic_iterations):
        #         d_loss = discriminator_training_iteration(
        #             x,
        #             discriminator,
        #             generator,
        #             d_optimizer,
        #             z_size,
        #             mode_z,
        #             lambda_gp,
        #             device,
        #         )
        #     d_losses.append(d_loss)
        #     g_losses.append(
        #         generator_training_iteration(
        #             x, generator, discriminator, g_optimizer, z_size, mode_z, device
        #         )
        #     )
        # # print("Epoch number: ", epoch, "dloss:", d_losses[-1], "gloss:", g_losses[-1], end='\r')

        # print(f"Epoch {epoch:03d} | D Loss >>" f" {torch.FloatTensor(d_losses).mean():.4f}")
        # generator.eval()
        # epoch_samples_wgan.append(
        #     create_samples(generator, fixed_z, batch_size, (1, 28, 28))
        #     .detach()
        #     .cpu()
        #     .numpy(),
        # )
        # np.save(
        #     f"/home/carlfre/uni/wasserstein/wasserstein_rashka/output/generated_{epoch}.npy",
        #     epoch_samples_wgan[-1],
        # )
        # # Save the weights of generator and discriminator as checkpoints
        # torch.save(
        #     generator.state_dict(),
        #     f"/home/carlfre/uni/wasserstein/wasserstein_rashka/checkpoints/gen_model_epoch_{epoch}.pth",
        # )
        # torch.save(
        #     discriminator.state_dict(),
        #     f"/home/carlfre/uni/wasserstein/wasserstein_rashka/checkpoints/disc_model_epoch_{epoch}.pth",
        # )

    # np.save('/home/carlfre/uni/wasserstein/wasserstein_rashka/generated_samples.npy', epoch_samples_wgan[-1])


    # Save the generated samples as a tensor
    # samples.save('/home/carlfre/uni/wasserstein/wasserstein_rashka/generated_samples.pt')
    # torch.save(torch.tensor(epoch_samples_wgan), '/home/carlfre/uni/wasserstein/wasserstein_rashka/generated_samples.pt')
    # print("Generated samples saved.")
