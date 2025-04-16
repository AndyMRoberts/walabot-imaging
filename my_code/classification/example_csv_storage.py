import numpy as np

images = np.random.rand(100, 2048, 40).astype(np.float32)  # example data
images_flat = images.reshape(100, -1)  # shape: [100, 81920]
np.savetxt('images.csv', images_flat, delimiter=',')