import os
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torchvision.transforms as T
from torchmetrics import JaccardIndex
from torchvision import datasets, models, transforms

from Mammolibs import MMmodels, MMdataset, MMutils

parser = argparse.ArgumentParser()
parser.add_argument('--it', type=int, required=False)
parser.add_argument('--lr', type=float, required=False)
args = parser.parse_args()


def train_resnet(index, state_dict, dataset, lr=1e-3, pre_iter=0, niters=10,
             batch_size=16, current_dir='/home'):

    device = xm.xla_device()

    model = models.vgg19(weights='DEFAULT')
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 1)   
    model = model.to(device).train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.BCEWithLogitsLoss()
    # jaccard = JaccardIndex(num_classes=2, task='binary')
    criterion = nn.BCEWithLogitsLoss()
    
    loss = 100

    for it in range(pre_iter+1, pre_iter+niters+1):
        start = time.time()
        dataset.shuffle()
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
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device) # noqa
        batch_loss = [100]
        for batch_no, batch in enumerate(para_train_loader): # noqa
            if index == 0 and batch_no == 0:
                batch_loss = []
            patches, labels, images, rois = batch
            logits = model(patches)
            train_loss = criterion(logits, labels)
            optimizer.zero_grad()
            train_loss.backward()
            xm.optimizer_step(optimizer)
            loss = train_loss.cpu()
            reduced_loss = xm.mesh_reduce('loss_reduce', loss, lambda x: sum(x) / len(x))  # Reduce the loss values from all TPU cores
            loss = reduced_loss.item()  # Convert reduced loss from tensor to Python scalar
            if index == 0:
                batch_loss.append(loss)
            if index == 0 and batch_no % 10 == 0:
                print("Batch:{:4d}  |  Iter:{:4d}  |  Tr_loss: {:.4f}".format( # noqa
                batch_no, it, loss)) # noqa
        if index == 0:
            print("=======================================================================") # noqa
            print("Master Process  |  Iter:{:4d}  |  Tr_loss: {:.4f}  |  Time: {}".format( # noqa
            it, np.array(batch_loss).mean(), MMutils.convert_seconds_to_time(time.time()-start))) # noqa
            print("=======================================================================") # noqa
    if index == 0:
        MMutils.save_model(model.cpu(), current_dir, pre_iter+niters)


if __name__ == '__main__':
    print('Training Resnet...')

    # model = MMmodels.UNet()
    # print(f'Total trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}') # noqa
    print(f'Total trainable parameters = {7738742}') # noqa
    
    current_dir = os.path.dirname(os.path.realpath(__file__))

    gcs_path = 'gs://last_dataset/labelled-dataset/BreastMammography/Benign/'
    dataset = MMdataset.MMImageSet(gcs_path, stage='local', aug=True)

    n_iter = 20
    if args.it:
        n_iter = args.it

    lr = 3e-3
    if args.lr:
        lr = args.lr
    
    xmp.spawn(train_resnet, args=(
        None,     # model
        dataset,        # dataset
        3e-3,             # lr
        0,       # pre_iter
        n_iter,         # niters
        128,            # batch_size
        current_dir     # current_dir
        ), start_method='forkserver')
    