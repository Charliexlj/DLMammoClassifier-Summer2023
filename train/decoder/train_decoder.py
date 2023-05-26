
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
import os
import time
import rarfile
import argparse

from Mammolibs import models as MMmodels
from Mammolibs import dataset as MMdataset

parser = argparse.ArgumentParser()
parser.add_argument('--ori', type=str, required=True)
parser.add_argument('--dest', type=str, required=True)
args = parser.parse_args()

# Specify the path to the RAR file and the destination directory
rar_path = args.ori
extract_dir = args.dest

os.makedirs(extract_dir, exist_ok=True)

# Open the RAR file and extract its contents to the destination directory
with rarfile.RarFile(rar_path) as rf:
    rf.extractall(extract_dir)


class Pretrain_Encoder(nn.Module):
    def __init__(self, in_channel=1, out_vector=1024, num_filter=32):
        super(Pretrain_Encoder, self).__init__()
        self.encoder = MMmodels.Mammo_Encoder(in_channel=in_channel, base_channel=num_filter)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32*32*512, 4096),
            nn.ReLU()
            )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
            )
        self.fc3 = nn.Linear(4096, out_vector)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def mutations(image):
    image1, image2 = T.RandomRotation(180)(image), T.RandomRotation(180)(image)
    image1, image2 = T.RandomAutocontrast()(image1), T.RandomAutocontrast()(image2)
    return image1, image2


def train_encoder(model, dataset, lr=1e-3, num_epochs=1000, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Device: {0}'.format(device))
    model.to(device)

    def NT_Xent_loss(a, b):
        tau = 1
        a_norm = torch.norm(a, dim=1).reshape(-1, 1)
        a_cap = torch.div(a, a_norm)
        b_norm = torch.norm(b, dim=1).reshape(-1, 1)
        b_cap = torch.div(b, b_norm)
        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
        a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
        b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
        sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
        sim_by_tau = torch.div(sim, tau)
        exp_sim_by_tau = torch.exp(sim_by_tau)
        sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
        exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
        numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap,b_cap_a_cap),tau))
        denominators = sum_of_rows - exp_sim_by_tau_diag
        num_by_den = torch.div(numerators, denominators)
        neglog_num_by_den = -torch.log(num_by_den)
        return torch.mean(neglog_num_by_den)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs+1):
        if epoch % 10 == 1:
            start = time.time()
        model.train()
        images, _ = dataset.get_tr_random_batch(batch_size)
        images = torch.from_numpy(np.array(images))
        images1, images2 = mutations(images)
        images1, images2 = images1.to(device, dtype=torch.float32), images2.to(device, dtype=torch.float32)
        logits1, logits2 = model(images1), model(images2)

        optimizer.zero_grad()
        train_loss = NT_Xent_loss(logits1, logits2)
        train_loss.backward()
        optimizer.step()
        end = time.time()
        if epoch % 10 == 0:
            model.eval()
            # Disabling gradient calculation during reference to reduce memory consumption
            with torch.no_grad():
                # Evaluate on a batch of test images and print out the test loss
                test_images, _ = dataset.get_tr_random_batch(32)
                test_images = torch.from_numpy(np.array(test_images))
                test_images1, test_images2 = mutations(test_images)
                test_images1, test_images2 = test_images1.to(device, dtype=torch.float32), test_images2.to(device, dtype=torch.float32)
                test_logits1, test_logits2 = model(test_images1), model(test_images2)
                test_loss = NT_Xent_loss(test_logits1, test_logits2)
                print("Iter:{:5d}  |  Tr_loss: {:.4f}  |  Val_loss: {:.4f}  |  Time per 10 iter: {:.4f}s".format(epoch, train_loss, test_loss, end-start))
    return model


if __name__ == '__main__':
    print('Enter script main function')
    model = Pretrain_Encoder()
    print(model)
    print(f'Total trainable parameters = {sum(p.numel() for p in model.parameters())}')

    benign_path = '/home/DLMammoClassifier-Summer2023/Dataset of Mammography with Benign Malignant Breast Masses/INbreast+MIAS+DDSM Dataset/Benign Masses/'
    malignant_path = '/home/DLMammoClassifier-Summer2023/Dataset of Mammography with Benign Malignant Breast Masses/INbreast+MIAS+DDSM Dataset/Malignant Masses/'

    image_paths = [benign_path, malignant_path]
    dataset = MMdataset.BreastImageSet(image_paths)

    num_epochs = 10000
    batch_size = 32

    trained_model = train_encoder(model, dataset, lr=1e-3, num_epochs=num_epochs, batch_size=batch_size)
