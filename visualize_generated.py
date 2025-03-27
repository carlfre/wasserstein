import numpy as np
import matplotlib.pyplot as plt


generated = np.load(f"output/generated_{2}.npy")
for i in range(1, 20):
    im = generated[i, 0, :, :]
    plt.imshow(im, cmap='gray')
    plt.show()

    
