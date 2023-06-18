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

from Mammolibs import MMmodels, MMdataset, MMutils

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=str, required=True)
parser.add_argument('--it', type=int, required=False)
parser.add_argument('--lr', type=float, required=False)
parser.add_argument('--autoencoder', type=str, required=False)
args = parser.parse_args()


def finetune(index, state_dict, dataset, lr=1e-3, pre_iter=0, niters=10,
                  batch_size=16, current_dir='/home'):

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

    model = MMmodels.UNet()
    if state_dict:
        unet_state_dict = model.state_dict()
        partial_state_dict = {k: v for k, v in state_dict.items() if k in unet_state_dict and v.size() == unet_state_dict[k].size()}
        unet_state_dict.update(partial_state_dict)
        model.load_state_dict(unet_state_dict)
    model = model.to(device).train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    loss = 100
    for it in range(pre_iter+1, pre_iter+niters+1):
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device) # noqa
        start = time.time()
        for batch_no, batch in enumerate(para_train_loader): # noqa
            if index==0:
                print(f'start batch {batch_no}...')
            images, labels = batch
            if index==0:
                print('images shape: ', images.shape)
                print('labels shape: ', labels.shape)
                image_labels = torch.stack((images, labels), dim=1).unsqueeze(2)
                print('image_labels shape: ', image_labels.shape)
                print('image_label shape: ', image_labels[0].shape)
            '''
            images = torch.stack([MMdataset.mutations(image_label) for image_label in zip(images,labels)])
            
            logits = model(images)
            train_loss = criterion(logits, labels)
            optimizer.zero_grad()
            train_loss.backward()
            xm.optimizer_step(optimizer)
            loss = train_loss.cpu()
        '''
        if index == 0:
            print("Master Process  |  Iter:{:4d}  |  Tr_loss: {:.4f}  |  Time: {}".format( # noqa
            it, loss.item(), MMutils.convert_seconds_to_time(time.time()-start))) # noqa

    if index == 0:
        MMutils.save_model(model.cpu(), current_dir, pre_iter+niters)


if __name__ == '__main__':
    print('Finetuning...')

    # model = MMmodels.UNet()
    # print(f'Total trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}') # noqa
    print(f'Total trainable parameters = {7738742}') # noqa
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if args.pretrain == 'no':
        pre_iter = 0
        state_dict = torch.load(f'{current_dir}/autoencoder/model_iter_{int(args.autoencoder)}.pth') # noqa
        print(f'Find model weights at {current_dir}/autoencoder/model_iter_{int(args.autoencoder)}.pth, loading...') # noqa
    else:
        pre_iter = int(args.pretrain)
        state_dict = torch.load(f'{current_dir}/model_iter_{pre_iter}.pth') # noqa
        print(f'Find model weights at {current_dir}/model_iter_{pre_iter}.pth, loading...') # noqa

    gcs_path = 'gs://combined-dataset/labelled-dataset/CombinedBreastMammography/'
    dataset = MMdataset.MMImageSet(gcs_path, stage='finetune', aug=True)

    n_iter = 20
    if args.it:
        n_iter = args.it

    lr = 3e-3
    if args.lr:
        lr = args.lr
    
    xmp.spawn(finetune, args=(
        state_dict,     # model
        dataset,        # dataset
        lr,             # lr
        pre_iter,       # pre_iter
        n_iter,         # niters
        64,             # batch_size
        current_dir     # current_dir
        ), start_method='forkserver')
    