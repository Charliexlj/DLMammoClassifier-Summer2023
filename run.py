import os

import matplotlib.pyplot as plt
import numpy as np
import random

import torch

import gcsfs
import imageio
import torchvision.transforms as T

from Mammolibs import MMmodels

import cv2
import torchvision
import torch.nn as nn

import sys
sys.path.append('/home/DLMammoClassifier-Summer2023')

import resnet.classification.svm_clf as clf


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        img_modules = list(models.children())[:-1]
        self.ModelA = nn.Sequential(*img_modules)
        self.relu = nn.ReLU()
        self.Linear3 = nn.Linear(512, 2, bias = True)

    def forward(self, x):
        x = self.ModelA(x) # N x 1024 x 1 x 1
        x1 = torch.flatten(x, 1) 
        x2 = self.Linear3(x1)
        return  x1, x2


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


def crop_center(img, cx, cy, size):
    """
    Crop out a square patch from an image.
    cx, cy are the center of the patch.
    """
    start_x = max(cx - size // 2, 0)
    if start_x + size > img.size(-1):
        start_x = img.size(-1) - size

    start_y = max(cy - size // 2, 0)
    if start_y + size > img.size(-2):
        start_y = img.size(-2) - size

    return img[..., start_y:start_y+size, start_x:start_x+size]


def process_images(images, labels, size):
    patches = []

    for img, lbl in zip(images, labels):
        lbl = lbl.squeeze(0)
        # The center of the label '1'
        y, x = (lbl == 1).nonzero().float().mean(0)

        # Crop a patch from the image
        patch = crop_center(img, x, y, size)

        patches.append(patch)

    # Convert list of tensors into a 4D tensor
    patches = torch.stack(patches)

    return patches

load_model = 'resnet/model_ResNet18.pt'

models = torchvision.models.resnet18(pretrained=True)

net = MyModel()
params = net.parameters()
optimizer=torch.optim.Adam(net.parameters())

if os.path.exists(load_model):
    checkpoint=torch.load(load_model,map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("model loaded successfully")
    print('starting training after epoch: ',checkpoint['epoch'])
    loaded_flag=True


def get_features(patch):
    patch = patch.numpy()
    net = MyModel()
    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = torch.tensor(img_arr)
    img_arr = img_arr.permute(2,0,1)
    img = img_arr.unsqueeze(0)
    net = net.cpu()
    X,_ = net(img)
    X = X.cpu().detach()
    return X

if __name__ == '__main__':
    model = MMmodels.UNet()
    iter = input("Model iter: ")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    state_dict = torch.load(f'{current_dir}/train/unet/model_iter_{iter}.pth') # noqa
    model.load_state_dict(state_dict)
    print(f'Find model weights at {current_dir}/train/unet/model_iter_{iter}.pth, loading...') # noqa

    # gcs_path = 'gs://last_dataset/labelled-dataset/BreastMammography/Benign/' # noqa
    # gcs_path2 = gcs_path.replace('Benign', 'Malignant')
    fs = gcsfs.GCSFileSystem()

    # filenames = [s for s in fs.ls(gcs_path) if s.endswith(('.png', '.jpg', '.jpeg'))] + \
    # [s for s in fs.ls(gcs_path2) if s.endswith(('.png', '.jpg', '.jpeg'))] # noqa
    # labels_names = [filename.replace('BreastMammography', 'ROIMask').replace("MAMMO", "ROI", 1) for filename in filenames] # noqa
    # print(f'The dataset contain {len(filenames)} images...')
    
    # idx = [random.randint(0, len(labels_names)-1) for _ in range(12)]

    # images = read_images(filenames, idx)
    # roi = read_images(labels_names, idx)
    
    img_path = 'gs://' + input('Image path:')
    roi_path = img_path.replace('BreastMammography', 'ROIMask').replace("MAMMO", "ROI", 1)
    image = read_image(img_path)
    roi = read_image(roi_path)
    
    roi = np.array(roi).reshape((256, 256))
    roi = np.where(roi >= 0.5, 1, 0)
    roi = T.ToTensor()(roi)

    logits = model(image.unsqueeze(0))
    logits = torch.argmax(logits, dim=1).detach().unsqueeze(0)
    
    patches = process_images(image.unsqueeze(0), logits, size=56)
    
    feature = get_features(patches[0])
    print(feature.shape)
    
    pred = clf.predict(feature)
    print(pred)

    # np_roi = np.array(roi).reshape((4, 256, 256))
    # np_roi = np.where(np_roi >= 0.5, 1, 0)
    
    # roi_test = T.ToTensor()(np_roi)
    # print("roi unique: ", torch.unique(roi_test))

    # logits_np = torch.argmax(logits, dim=1).detach().numpy()
    # logits_test = T.ToTensor()(logits_np)
    # print("logits unique: ", torch.unique(logits_test))

    # image_np = image.numpy()
    # roi_np = roi.numpy()

    # fig, axs = plt.subplots(3, 4, figsize=(12, 9))

    # for i in range(4):
    #     # plot image
    #     axs[0, i].imshow(image_np[i][0], cmap='gray')
    #     axs[0, i].set_title(f'Image {i+1}')
    #     axs[0, i].axis('off')
        
    #     # plot roi
    #     axs[1, i].imshow(roi_np[i][0], cmap='gray')
    #     axs[1, i].set_title(f'Label {i+1}')
    #     axs[1, i].axis('off')
        
    #     # plot logits
    #     axs[2, i].imshow(logits_np[i], cmap='gray')
    #     axs[2, i].set_title(f'Logit {i+1}')
    #     axs[2, i].axis('off')

    # plt.tight_layout()
    # plt.savefig('plot.png')
