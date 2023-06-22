import gcsfs
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


class MMImageSet(Dataset):
    def __init__(self, gcs_path, stage='encoder', aug=True):
        super(MMImageSet, self).__init__()
        self.fs = gcsfs.GCSFileSystem()
        '''https://storage.cloud.google.com/last_dataset/labelled-dataset/BreastMammography/Benign/MDB_MAMMO_104_0_0.jpg'''
        self.stage = stage
        if self.stage == 'finetune':
            gcs_path2 = gcs_path.replace('Benign', 'Malignant')
            self.filenames = [s for s in self.fs.ls(gcs_path) if s.endswith(('.png', '.jpg', '.jpeg'))] + \
            [s for s in self.fs.ls(gcs_path2) if s.endswith(('.png', '.jpg', '.jpeg'))] # noqa
            print(f'The dataset contain {len(self.filenames)} images...')
            self.labels = [filename.replace('BreastMammography', 'ROIMask').replace("_MAMMO_", "_ROI_", 1) for filename in self.filenames] # noqa
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
            return image
        else:
            roi = self.read_image(self.labels[idx])
            roi = np.array(roi).reshape((256, 256))
            roi = np.where(roi >= 0.5, 1, 0)
            roi = T.ToTensor()(roi)
            return image, roi
        
    def shuffle(self):
        indices = list(range(len(self.filenames)))
        random.shuffle(indices)
        self.filenames = [self.filenames[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
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