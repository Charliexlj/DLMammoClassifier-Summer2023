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
    
    model_1 = MMmodels.UNet()
    iter = '60'
    current_dir = os.path.dirname(os.path.realpath(__file__))
    state_dict = torch.load(f'{current_dir}/train/unet/model_iter_{iter}.pth') # noqa
    model_1.load_state_dict(state_dict)
    print(f'Find model weights at {current_dir}/train/unet/model_iter_{iter}.pth, loading...') # noqa

    gcs_path = 'gs://last_dataset/labelled-dataset/BreastMammography/Benign/' # noqa
    gcs_path2 = gcs_path.replace('Benign', 'Malignant')
    fs = gcsfs.GCSFileSystem()

    filenames = [s for s in fs.ls(gcs_path) if s.endswith(('.png', '.jpg', '.jpeg'))] + \
    [s for s in fs.ls(gcs_path2) if s.endswith(('.png', '.jpg', '.jpeg'))] # noqa
    labels_names = [filename.replace('BreastMammography', 'ROIMask').replace("MAMMO", "ROI", 1) for filename in filenames] # noqa
    print(f'The dataset contain {len(filenames)} images...')

    idx = [random.randint(0, len(labels_names)-1) for _ in range(12)]

    images = read_images(filenames, idx)
    labels = read_images(labels_names, idx)

    logits_1 = model_1(images)

    images_np = images.numpy().reshape(12, 256, 256)*255
    labels_np = labels.numpy().reshape(12, 256, 256).astype(np.uint8)
    logits_np_1 = torch.argmax(logits_1, dim=1).detach().numpy().reshape(12, 256, 256).astype(np.uint8)

    # Create a blank array to hold the highlighted images
    highlighted_images_1 = np.zeros((12, 256, 256, 3), dtype=np.uint8)

    for i in range(12):
        original_image_gray = images_np[i]  # Grayscale original image
        
        # Convert the original grayscale image to three channels
        original_image_rgb = cv2.cvtColor(original_image_gray, cv2.COLOR_GRAY2RGB)
        
        # Create a mask by thresholding the segmentation image
        _, binary_mask_t = cv2.threshold(labels_np[i], 0, 255, cv2.THRESH_BINARY)
        _, binary_mask_p = cv2.threshold(logits_np_1[i], 0, 255, cv2.THRESH_BINARY)
        
        # Convert the binary mask to a colored overlay (e.g., green)
        overlay_color_t = (0, 255, 0)  # Green color
        overlay_t = np.zeros_like(original_image_rgb, dtype=np.uint8)
        overlay_t[np.where(binary_mask_t > 0)] = overlay_color_t
        overlay_t = overlay_t.astype(original_image_rgb.dtype)
        
        overlay_color_p = (0, 0, 255)  # Blue color
        overlay_p = np.zeros_like(original_image_rgb, dtype=np.uint8)
        overlay_p[np.where(binary_mask_p > 0)] = overlay_color_p
        overlay_p = overlay_p.astype(original_image_rgb.dtype)

        # Combine the overlay with the original image
        highlighted_image_1 = cv2.addWeighted(overlay_t, 1, overlay_p, 1, 0)
        highlighted_image_2_1 = cv2.addWeighted(original_image_rgb, 0.7, highlighted_image_1, 0.3, 0)
        
        # Store the highlighted image in the array
        highlighted_images_1[i] = highlighted_image_2_1
        
    fig, axs = plt.subplots(3, 4, figsize=(24, 15))
    
    names = [filenames[i] for i in idx]

    for i in range(3):
        for j in range(4):
            
            s = names[i*4+j]
            parts = s.split('/')
            result = '/'.join(parts[3:])
            
            axs[i, j].imshow(highlighted_images_1[i*4+j])
            axs[i, j].set_title(f'{result}')
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('plot.png')
    plt.show()

    print('Saved plot.png')
