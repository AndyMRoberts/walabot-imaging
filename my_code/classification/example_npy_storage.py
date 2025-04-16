import numpy as np

# ---- Configuration ----
num_samples = 100
image_height = 2048
image_width = 40
num_classes = 4

# ---- Generate dummy image data ----
# Random float32 values between 0 and 1, simulating grayscale images
images = np.random.rand(num_samples, image_height, image_width).astype(np.float32)

# ---- Generate dummy labels ----
# Random integers from 0 to 3 for class labels
labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)

# ---- Save as .npy ----
np.save('images.npy', images)
np.save('labels.npy', labels)

print(f"Saved images of shape {images.shape} to 'images.npy'")
print(f"Saved labels of shape {labels.shape} to 'labels.npy'")
