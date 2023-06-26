import cv2
import imageio
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
import json


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Encoder(nn.Module):
    def __init__(self, in_channel=1, base_channel=32):
        super(Encoder, self).__init__()
        # 256*256*1
        n = base_channel
        self.conv1 = conv_block(1, n)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128*128*32
        n = n*2
        self.conv2 = conv_block(n//2, n)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64*64*64
        n = n*2
        self.conv3 = conv_block(n//2, n)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32*32*128
        n = n*2
        self.conv4 = conv_block(n//2, n)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 16*16*256
        n = n*2
        self.conv5 = conv_block(n//2, n)
        # 16*16*512

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.pool1(x))
        x = self.conv3(self.pool2(x))
        x = self.conv4(self.pool3(x))
        x = self.conv5(self.pool4(x))
        return x


class Decoder(nn.Module):
    def __init__(self, in_channel=512, out_channel=2, mode="Autoencoder"):
        super(Decoder, self).__init__()
        # 16*16*512
        n = in_channel//2
        self.upconv4 = nn.ConvTranspose2d(n*2, n, kernel_size=2, stride=2)
        # 16*16*256
        if mode == "Autoencoder": self.conv4 = conv_block(n, n) # noqa
        else: self.conv4 = conv_block(n*2, n) # noqa
        # 32*32*128
        n = n//2
        self.upconv3 = nn.ConvTranspose2d(n*2, n, kernel_size=2, stride=2)
        if mode == "Autoencoder": self.conv3 = conv_block(n, n) # noqa
        else: self.conv3 = conv_block(n*2, n) # noqa
        # 64*64*64
        n = n//2
        self.upconv2 = nn.ConvTranspose2d(n*2, n, kernel_size=2, stride=2)
        if mode == "Autoencoder": self.conv2 = conv_block(n, n) # noqa
        else: self.conv2 = conv_block(n*2, n) # noqa
        # 128*128*32
        n = n//2
        self.upconv1 = nn.ConvTranspose2d(n*2, n, kernel_size=2, stride=2)
        if mode == "Autoencoder": self.conv1 = conv_block(n, out_channel) # noqa
        else: self.conv1 = conv_block(n*2, out_channel) # noqa
        # 256*256*2
        self.conv = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1) # noqa

    def forward(self, x):
        x = self.conv4(self.upconv4(x))
        x = self.conv3(self.upconv3(x))
        x = self.conv2(self.upconv2(x))
        x = self.conv1(self.upconv1(x))
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, num_filter=32):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channel=in_channel, base_channel=32)
        self.decoder = Decoder(in_channel=512, out_channel=out_channel, mode="UNet") # noqa

    def forward(self, x):
        enc1 = self.encoder.conv1(x)
        enc2 = self.encoder.conv2(self.encoder.pool1(enc1))
        enc3 = self.encoder.conv3(self.encoder.pool2(enc2))
        enc4 = self.encoder.conv4(self.encoder.pool3(enc3))

        bottleneck = self.encoder.conv5(self.encoder.pool4(enc4))

        dec4 = self.decoder.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder.conv4(dec4)
        dec3 = self.decoder.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder.conv3(dec3)
        dec2 = self.decoder.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder.conv2(dec2)
        dec1 = self.decoder.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder.conv1(dec1)
        return torch.sigmoid(self.decoder.conv(dec1))


unet_model = UNet()
# need to change to local
# need to change to local
# need to change to local
unet_state_dict = torch.load(f'/content/unet.pth')  # need to change to local
print(f'Find UNet model weights at /content/unet.pth, loading...')
unet_model.load_state_dict(unet_state_dict)

resnet_model = models.resnet18(weights='DEFAULT')
nr_filters = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(nr_filters, 1)
# need to change to local
# need to change to local
# need to change to local
resnet_state_dict = torch.load(f'/content/resnet.pth')  # need to change to local
print(f'Find ResNet model weights at /content/resnet.pth, loading...')
resnet_model.load_state_dict(resnet_state_dict)


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
    nonzero_coords = (lbl == 1).nonzero()
    if nonzero_coords.nelement() == 0:  # If no elements found, continue to next iteration
        x, y = 128, 128
    else:
        y, x = nonzero_coords.float().mean(0)
        x = round(x.item())
        y = round(y.item())

    # Crop a patch from the image
    patch = crop_center(images, x, y, size)

    return patch


if __name__ == '__main__':
    Mammo = imageio.imread('/content/labelled-dataset_BreastMammography_Benign_CBIS_MAMMO-R-MLO_464_60_1.jpg')
    Mammo = torch.tensor(Mammo).unsqueeze(0).unsqueeze(0)
    # Mammo.shape = (1, 1, 256, 256)
    
    Seg = unet_model(Mammo.float())
    # Seg.shape = (1, 2, 256, 256)
    Seg_np = torch.argmax(Seg, dim=1).detach().numpy().reshape(1, 256, 256)
    # Seg_np.shape = (1, 256, 256)
    
    Patch = process_images_patch(Mammo.squeeze(0), torch.tensor(Seg_np), 56)
    # Patch.shape = (1, 56, 56)
    
    Patch = Patch.permute(1, 2, 0).numpy()
    Patch = cv2.resize(Patch, (224, 224))
    Patch_np = Patch.copy()
    Patch = torch.tensor(Patch).unsqueeze(0)
    Patch = Patch.repeat(3, 1, 1)
    # Patch.shape = (3, 224, 224)
    
    # Save Seg_np
    Seg_np_scaled = (Seg_np * 255).astype(np.uint8)  # Convert to 0-255 scale
    imageio.imsave('Seg_np.jpg', Seg_np_scaled.squeeze())  # Remove any singleton dimensions and save as .jpg

    # Save Patch_np
    Patch_np_scaled = (Patch_np * 255).astype(np.uint8)  # Convert to 0-255 scale
    imageio.imsave('Patch_np.jpg', Patch_np_scaled)  # Save as .jpg
    
    Flag = 'Benign'
    Label = resnet_model(Patch.unsqueeze(0).float())
    if Label.detach()[0] > 0:
        Flag = 'Malignant'
        
    with open('Flag.json', 'w') as file:
        json.dump({'Flag': Flag}, file)