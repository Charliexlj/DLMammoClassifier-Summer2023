import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
import os
import sys
import time
import rarfile
import argparse
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from Mammolibs import MMmodels, MMdataset, MMutils # noqa

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
        self.encoder = MMmodels.Mammo_Encoder(
            in_channel=in_channel, base_channel=num_filter)
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


def train_encoder(index, model, dataset, lr=1e-3, num_epochs=1000,
                  batch_size=16, save_path='/home'):

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=8,
        drop_last=True)

    device = xm.xla_device()
    print('Training with Device: {0}...'.format(device))
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
        numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()
                                         (a_cap_b_cap, b_cap_a_cap), tau))
        denominators = sum_of_rows - exp_sim_by_tau_diag
        num_by_den = torch.div(numerators, denominators)
        neglog_num_by_den = -torch.log(num_by_den)
        return torch.mean(neglog_num_by_den)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs+1):
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device) # noqa
        for batch in para_train_loader:
            # if epoch % 1 == 0:
            #     start = time.time()
            model.train()
            images1, images2 = batch
            logits1, logits2 = model(images1), model(images2)

            optimizer.zero_grad()
            train_loss = NT_Xent_loss(logits1, logits2)
            train_loss.backward()
            xm.optimizer_step(optimizer)
            print(f'epoch: {epoch}, train_loss{train_loss.cpu()}')
            '''
            model.eval()
            with torch.no_grad():
                MMutils.print_iteration_stats(epoch, train_loss, train_loss, 1, time.time()-start) # noqa

            if epoch % 200 == 0:
                MMutils.save_model(model, save_path, epoch)

            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    test_images1, test_images2 = batch
                    test_logits1 = model(test_images1)
                    test_logits2 = model(test_images2)
                    test_loss = NT_Xent_loss(test_logits1, test_logits2)
                    MMutils.print_iteration_stats(epoch, train_loss, test_loss, 1, time.time()-start) # noqa
            '''
    return model


if __name__ == '__main__':
    print('Training Encoder...')
    model = Pretrain_Encoder()
    # print(model)
    print('Total trainable parameters = '
          f'{sum(p.numel() for p in model.parameters())}')

    benign_path = '/home/DLMammoClassifier-Summer2023/Dataset of Mammography with Benign Malignant Breast Masses/INbreast+MIAS+DDSM Dataset/Benign Masses/' # noqa
    malignant_path = '/home/DLMammoClassifier-Summer2023/Dataset of Mammography with Benign Malignant Breast Masses/INbreast+MIAS+DDSM Dataset/Malignant Masses/' # noqa

    dataset = MMdataset.BreastImageSet([benign_path, malignant_path])

    trained_model = xmp.spawn(train_encoder, args=(model, dataset, 1e-3, 10, 32, os.path.dirname(os.path.realpath(__file__))), start_method='fork', nprocs=8) # noqa
    '''
    trained_model = train_encoder(
        model,
        dataset,
        lr=1e-3,
        num_epochs=10000,
        batch_size=32,
        save_path=os.path.dirname(os.path.realpath(__file__))
        )
    '''
