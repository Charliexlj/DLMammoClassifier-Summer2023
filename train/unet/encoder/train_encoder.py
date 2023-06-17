import os
import time
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torchvision.transforms as T
from pytorch_metric_learning import losses

from Mammolibs import MMmodels, MMdataset, MMutils

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=str, required=True)
parser.add_argument('--it', type=int, required=False)
parser.add_argument('--lr', type=float, required=False)
parser.add_argument('-test', action='store_true', required=False)
args = parser.parse_args()


def train_encoder(index, state_dict, dataset, lr=1e-3, pre_iter=0, niters=10,
                  batch_size=16, current_dir='/home', test_flag=False):
    if index == 0:
        print('start main training function...')

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

    model = MMmodels.Pretrain_Encoder()
    if state_dict:
        model.load_state_dict(state_dict)
    model = model.to(device).train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = losses.NTXentLoss(temperature=0.05)
    
    labels = torch.cat([torch.tensor([0]*(batch_size+1)), torch.arange(1, batch_size)], dim=0)
    
    loss = 100
    if index == 0:
        print('start main training loop...')
    for it in range(pre_iter+1, pre_iter+niters+1):
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device) # noqa
        try:
            batch_no, batch = next(iter(para_train_loader))
            print("Loaded a batch successfully")
        except Exception as e:
            print("Failed to load a batch. Error: ", e)
        if index == 0:
            print('para_train_loader finished...')
        start = time.time()
        for batch_no, batch in enumerate(para_train_loader): # noqa
            if index == 0:
                print(f'enter batch {batch_no}...')
            images = batch
            image0 = images[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)
            if index==0 and batch_no==0:
                print('images shape: ', images.shape)
                print('image0 shape: ', image0.shape)
            images = torch.cat([image0, images], dim=0)
            images = torch.stack([MMdataset.mutations(image) for image in images])
            if index==0 and batch_no==0:
                print('combined images shape: ', images.shape)
            
            logits = model(images)
            train_loss = criterion(logits, labels)
            optimizer.zero_grad()
            train_loss.backward()
            xm.optimizer_step(optimizer)
            loss = train_loss.cpu()
            if test_flag:
                break
        if index == 0:
            print("Master Process  |  Iter:{:4d}  |  Tr_loss: {:.4f}  |  Time: {}".format( # noqa
            it, loss.item(), MMutils.convert_seconds_to_time(time.time()-start))) # noqa

    if index == 0:
        MMutils.save_model(model.cpu(), current_dir, pre_iter+niters)


if __name__ == '__main__':
    print('Training Encoder...')
    '''
    model = MMmodels.Pretrain_Encoder()
    print(f'Total trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}') # noqa
    '''
    print(f'Total trainable parameters = {279447648}') # noqa

    current_dir = os.path.dirname(os.path.realpath(__file__))

    pre_iter = 0
    state_dict = None
    if args.pretrain != 'no':
        pre_iter = int(args.pretrain)
        state_dict = torch.load(f'{current_dir}/model_iter_{pre_iter}.pth') # noqa
        print(f'Find model weights at {current_dir}/model_iter_{pre_iter}.pth, loading...') # noqa
        print(f'Now start to train from iter {pre_iter}...')

    gcs_path = 'gs://combined-dataset/unlabelled-dataset/CombinedBreastMammography/'
    dataset = MMdataset.MMImageSet(gcs_path, aug=False)

    n_iter = 20
    if args.it:
        n_iter = args.it

    lr = 3e-3
    if args.lr:
        lr = args.lr

    xmp.spawn(train_encoder, args=(
        state_dict,     # model
        dataset,        # dataset
        lr,             # lr
        pre_iter,       # pre_iter
        n_iter,         # niters
        64,             # batch_size
        current_dir,     # current_dir
        args.test
        ), start_method='forkserver')
