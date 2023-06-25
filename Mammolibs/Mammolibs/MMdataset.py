import gcsfs
import cv2
import imageio
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T


def mutations(image):
    image = T.RandomRotation(180)(image)
    image = T.RandomAutocontrast()(image)
    image = T.RandomPerspective()(image)
    return image

def crop_center(img, cx, cy, size):
    """
    Crop out a square patch from an image.
    cx, cy are the center of the patch.
    """
    # image 1,256,256
    start_x = max(cx - size // 2, 0)
    if start_x + size > img.size(-1):
        start_x = img.size(-1) - size

    start_y = max(cy - size // 2, 0)
    if start_y + size > img.size(-2):
        start_y = img.size(-2) - size

    return img[..., start_y:start_y+size, start_x:start_x+size]


def process_images_patch(images, labels, size):
    # roi 1,256,256
    # image 1,256,256
    lbl = labels.squeeze(0)
    label = 0
    nonzero_coords = (lbl == 1).nonzero()
    if nonzero_coords.nelement() == 0:  # If no elements found, continue to next iteration
        x, y = 128,128
        label = 1
    else:
        y, x = nonzero_coords.float().mean(0)
        x = round(x.item())
        y = round(y.item())

    # Crop a patch from the image
    patch = crop_center(images, x, y, size)

    return patch, label


class MMImageSet(Dataset):
    def __init__(self, gcs_path, stage='encoder', aug=True):
        super(MMImageSet, self).__init__()
        self.fs = gcsfs.GCSFileSystem()
        '''https://storage.cloud.google.com/last_dataset/labelled-dataset/BreastMammography/Benign/MDB_MAMMO_104_0_0.jpg'''
        self.stage = stage
        if self.stage == 'finetune' or self.stage == 'local':
            gcs_path2 = gcs_path.replace('Benign', 'Malignant')
            self.filenames = [s for s in self.fs.ls(gcs_path) if s.endswith(('.png', '.jpg', '.jpeg'))] + \
            [s for s in self.fs.ls(gcs_path2) if s.endswith(('.png', '.jpg', '.jpeg'))] # noqa
            self.labels = [filename.replace('BreastMammography', 'ROIMask').replace("MAMMO", "ROI", 1) for filename in self.filenames] # noqa
        else:
            if aug: self.filenames = [s for s in self.fs.ls(gcs_path) if s.endswith(('.png', '.jpg', '.jpeg'))] # noqa
            else: self.filenames = [s for s in self.fs.ls(gcs_path) if s.count('_') == 1 and s.endswith(('.png', '.jpg', '.jpeg'))] # noqa
        print(f'The dataset contain {len(self.filenames)} images...')

    def __len__(self):
        return len(self.filenames)
    
    def process_image(self, image):
        image = T.ToTensor()(image)
        image = image if image.shape[0] != 3 else image.mean(dim=0, keepdim=True)
        return image
    
    def read_image(self, path):
        with self.fs.open(path, 'rb') as f:
            try:
                image = imageio.imread(f)
                return self.process_image(image)
            except Exception as e:
                print(f"Error reading image from path {path}: {e}")
                return None

    def __getitem__(self, idx):
        image = self.read_image(self.filenames[idx])
        if image is None or not image.shape[0]==1:
            print(image.shape)
            print(f"Image at index {idx} is None. Returning a zero tensor instead.")
            return torch.zeros(1, 256, 256)
        if self.stage != 'finetune':
            if self.stage == 'local':
                filename = self.filenames[idx]
                roi = self.read_image(self.filenames[idx].replace('BreastMammography', 'ROIMask').replace("MAMMO", "ROI", 1))
                roi = np.array(roi).reshape((256, 256))
                roi = np.where(roi >= 0.5, 1, 0)
                roi = T.ToTensor()(roi)
                # roi 1,256,256
                # image 1,256,256
                patch, no_flag = process_images_patch(image, roi, size=56)
                # patch 1,256,256
                img_arr = patch.permute(1, 2, 0).numpy()
                img_arr = cv2.resize(img_arr, (224,224))
                img_arr = torch.tensor(img_arr).unsqueeze(0)
                img_arr = img_arr.repeat(3, 1, 1)
                if no_flag:
                    label = torch.tensor(0)
                elif 'Benign' in filename:
                    label = torch.tensor(0)
                else:
                    label = torch.tensor(1)
                return img_arr, label.float(), image, roi
            else:
                return image
        else:
            roi = self.read_image(self.filenames[idx].replace('BreastMammography', 'ROIMask').replace("MAMMO", "ROI", 1))
            roi = np.array(roi).reshape((256, 256))
            roi = np.where(roi >= 0.5, 1, 0)
            roi = T.ToTensor()(roi)
            return image, roi
        
    def shuffle(self):
        indices = list(range(len(self.filenames)))
        random.shuffle(indices)
        self.filenames = [self.filenames[i] for i in indices]
        # self.labels = [self.labels[i] for i in indices]
'''

def rgb_to_grayscale(img):
    return img.mean(dim=0, keepdim=True)


class MMImageSet(Dataset):
    def __init__(self, gcs_path, stage='encoder', aug=True):
        super(MMImageSet, self).__init__()
        self.fs = gcsfs.GCSFileSystem()
        if aug:
            self.filenames = self.fs.ls(gcs_path)
        else:
            self.filenames = [s for s in  self.fs.ls(gcs_path) if s.count('_') == 1]
        print(self.filenames[:10])
        self.stage = stage
        print(f'The dataset contain {len(self.filenames)} images...')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with self.fs.open(self.filenames[idx], 'rb') as f:
            image = imageio.imread(f)
        if image.shape[0] == 3:
            image = rgb_to_grayscale(image)
        image = T.ToTensor()(image)
        return image
'''