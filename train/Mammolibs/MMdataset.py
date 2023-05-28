import imageio
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import gcsfs
import torchvision.transforms as T
import numpy as np
import os
import cv2


def normalise_intensity(image, thres_roi=1.0):
    """ Normalise the image intensity by the mean and standard deviation """
    # ROI defines the image foreground
    image = cv2.resize(image, (256, 256))
    val_l = np.percentile(image, thres_roi)
    roi = (image >= val_l)
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    eps = 1e-6
    image2 = (image - mu) / (sigma + eps)
    return image2


def resize_256(image):
    return cv2.resize(image, (256, 256))


def mutations(image):
    image1, image2 = T.RandomRotation(180)(image), T.RandomRotation(180)(image)
    image1 = T.RandomAutocontrast()(image1)
    image2 = T.RandomAutocontrast()(image2)
    return image1, image2


class BreastImageSet(Dataset):
    """ Brain image set """
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.images = []

        image_names_ = []
        for image_path in image_paths:
            image_names_.append(sorted(os.listdir(image_path)))

        for i, image_names in enumerate(image_names_):
            for image_name in image_names:
                if image_name.endswith('.png'):
                    # Read the image
                    image = imageio.imread(
                        os.path.join(image_paths[i], image_name))
                    self.images += [image]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = normalise_intensity(self.images[idx])
        image = ToTensor()(image)
        return mutations(image)


class MMImageSet(Dataset):
    def __init__(self, gcs_path, stage='encoder'):
        super(MMImageSet, self).__init__()
        self.fs = gcsfs.GCSFileSystem()
        self.filenames = self.fs.ls(gcs_path)
        self.stage = stage
        print(f'The dataset contain {len(self.filenames)} images...')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with self.fs.open(self.filenames[idx], 'rb') as f:
            image = imageio.imread(f)
            print('Got an image...')
        image = ToTensor()(image)
        return mutations(image)
