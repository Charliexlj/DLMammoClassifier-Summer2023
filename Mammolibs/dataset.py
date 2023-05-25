import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import time
import os
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2

from sklearn.utils import shuffle

def normalise_intensity(image, thres_roi=1.0):
    """ Normalise the image intensity by the mean and standard deviation """
    # ROI defines the image foreground
    image = cv2.resize(image, (512, 512))
    val_l = np.percentile(image, thres_roi)
    roi = (image >= val_l)
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    eps = 1e-6
    image2 = (image - mu) / (sigma + eps)
    return image2

class BreastImageSet(Dataset):
    """ Brain image set """
    def __init__(self, image_paths, labels=[0,1]):
        self.image_paths = image_paths
        self.images = []
        self.labels = []

        image_names_ = []
        for image_path in image_paths:
            image_names_.append(sorted(os.listdir(image_path)))

        for i, image_names in enumerate(image_names_):
            for image_name in image_names:
                if image_name.endswith('.png'):
                # Read the image
                    image = imageio.imread(os.path.join(image_paths[i], image_name))
                    self.images += [image]
                    self.labels.append(labels[i])
        self.images, self.labels = shuffle(self.images, self.labels, random_state=101)
        self.labels = np.eye(len(labels))[self.labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = normalise_intensity(self.images[idx])
        label = self.labels[idx]
        return image, label

    def get_tr_random_batch(self, batch_size):
        images, labels = [], []
        indices = np.random.choice(int(len(self)*0.8), batch_size)
        for idx in indices:
            image, label = self[idx]
            image = np.expand_dims(image, axis=0)
            images.append(image)
            labels.append(label)
        return images, labels

    def get_val_random_batch(self, batch_size):
        images, labels = [], []
        indices = np.random.choice(int(len(self)*0.2), batch_size)
        for idx in indices:
            image, label = self[idx+int(len(self)*0.8)]
            image = np.expand_dims(image, axis=0)
            images.append(image)
            labels.append(label)
        return images, labels