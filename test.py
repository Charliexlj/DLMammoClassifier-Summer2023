import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

import torch

import gcsfs
import imageio
import torchvision.transforms as T

from Mammolibs import MMmodels


def process_image(image):
    image = T.ToTensor()(image)
    image = image if image.shape[0] != 3 else image.mean(dim=0, keepdim=True)
    return image


def read_image(path):
    with fs.open(path, 'rb') as f:
        try:
            image = imageio.imread(f)
            return process_image(image)
        except Exception as e:
            print(f"Error reading image from path {path}: {e}")
            return None


def read_images(paths, indices):
    return torch.stack([read_image(paths[i]) for i in indices])


if __name__ == '__main__':
    # model = MMmodels.Autoencoder()
    # iter = input("Model iter: ")
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # state_dict = torch.load(f'{current_dir}/train/unet/autoencoder/model_iter_{iter}.pth') # noqa
    # model.load_state_dict(state_dict)
    # print(f'Find model weights at {current_dir}/train/unet/autoencoder/model_iter_{iter}.pth, loading...') # noqa

    gcs_path = 'gs://last_dataset/labelled-dataset/BreastMammography/Benign/' # noqa
    gcs_path2 = gcs_path.replace('Benign', 'Malignant')
    fs = gcsfs.GCSFileSystem()

    filenames = [s for s in fs.ls(gcs_path) if s.endswith(('.png', '.jpg', '.jpeg'))] + \
    [s for s in self.fs.ls(gcs_path2) if s.endswith(('.png', '.jpg', '.jpeg'))] # noqa
    labels_names = [filename.replace('BreastMammography', 'ROIMask').replace("MAMMO", "ROI", 1) for filename in self.filenames] # noqa
    print(f'The dataset contain {len(filenames)} images...')

idx = [random.randint(0, len(labels_names)-1) for _ in range(36)]

images = read_images(filenames, idx)
labels = read_images(labels_names, idx)

images_np = images.numpy().reshape(36, 256, 256)
labels_np = labels.numpy().reshape(36, 256, 256)

# Create a blank array to hold the highlighted images
highlighted_images = np.zeros((36, 256, 256, 3), dtype=np.uint8)

for i in range(36):
    original_image_gray = images_np[i]  # Grayscale original image
    
    # Convert the original grayscale image to three channels
    original_image_rgb = cv2.cvtColor(original_image_gray, cv2.COLOR_GRAY2RGB)
    
    # Create a mask by thresholding the segmentation image
    _, binary_mask = cv2.threshold(labels_np[i], 0, 255, cv2.THRESH_BINARY)
    
    # Convert the binary mask to a colored overlay (e.g., green)
    overlay_color = (0, 255, 0)  # Green color
    overlay = np.zeros_like(original_image_rgb)
    overlay[np.where(binary_mask > 0)] = overlay_color
    
    # Combine the overlay with the original image
    highlighted_image = cv2.addWeighted(original_image_rgb, 1, overlay, 0.5, 0)
    
    # Store the highlighted image in the array
    highlighted_images[i] = highlighted_image

fig, axs = plt.subplots(6, 6, figsize=(24, 24))

for i in range(6):
    for j in range(6):
        axs[i, j].imshow(highlighted_images[i*6+j])
        axs[i, j].set_title(f'Image {i*6+j+1}')
        axs[i, j].axis('off')

plt.tight_layout()
plt.savefig('plot.png')
plt.show()

print('Saved plot.png')
