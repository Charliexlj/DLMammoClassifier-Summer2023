import gcsfs
import imageio
import torch
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
        if aug: self.filenames = self.fs.ls(gcs_path)
        else: self.filenames = [s for s in  self.fs.ls(gcs_path) if s.count('_') == 1 and s.endswith(('.png', '.jpg', '.jpeg'))]
        self.stage = stage
        if self.stage == 'finetune':
            self.labels = [filename.replace('CombinedBreastMammography', 'CombinedROIMask').replace("_", "_ROI_", 1) for filename in self.filenames] # noqa
        print(f'The dataset contain {len(self.filenames)} images...')
        print(f'filename: {self.filenames[:3]}')
        print(f'labels: {self.labels[:3]}')

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
            return image, roi
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