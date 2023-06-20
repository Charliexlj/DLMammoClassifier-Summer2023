import os

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
    model = MMmodels.UNet()
    iter = input("Model iter: ")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    state_dict = torch.load(f'{current_dir}/train/unet/model_iter_{iter}.pth') # noqa
    model.load_state_dict(state_dict)
    print(f'Find model weights at {current_dir}/train/unet/model_iter_{iter}.pth, loading...') # noqa

    gcs_path = 'gs://combined-dataset/labelled-dataset/CombinedBreastMammography/' # noqa
    fs = gcsfs.GCSFileSystem()
    filenames = [s for s in fs.ls(gcs_path) if s.endswith(('.png', '.jpg', '.jpeg'))] # noqa
    print(f'The dataset contain {len(filenames)} images...')
    labels = [filename.replace('CombinedBreastMammography', 'CombinedROIMask').replace("_", "_ROI_", 1) for filename in filenames] # noqa
    idx = random.randint(0, len(labels)-4)

    idx = [random.randint(0, len(labels)) for _ in range(4)]

    image = read_images(filenames, idx)
    roi = read_images(labels, idx)

    logits = model(image)

    np_roi = np.array(roi).reshape((4, 256, 256))
    np_roi = np.where(np_roi >= 0.5, 1, 0)
    
    roi = T.ToTensor()(np_roi)
    print("roi unique: ", torch.unique(roi))
    print("logits unique: ", torch.unique(logits))

    logits_np = torch.argmax(logits, dim=1).detach().numpy()

    image_np = image.numpy()
    roi_np = roi.numpy()

    fig, axs = plt.subplots(3, 4, figsize=(12, 9))

    for i in range(4):
        # plot image
        axs[0, i].imshow(image_np[i][0], cmap='gray')
        axs[0, i].set_title(f'Image {i+1}')
        axs[0, i].axis('off')
        
        # plot roi
        axs[1, i].imshow(roi_np[i][0], cmap='gray')
        axs[1, i].set_title(f'Label {i+1}')
        axs[1, i].axis('off')
        
        # plot logits
        axs[2, i].imshow(logits_np[i], cmap='gray')
        axs[2, i].set_title(f'Logit {i+1}')
        axs[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('plot.png')
