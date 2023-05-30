import os
import time
import argparse

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from pytorch_metric_learning import losses

from Mammolibs import MMmodels, MMdataset, MMutils

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=str, required=True)
parser.add_argument('--it', type=int, required=False)
parser.add_argument('--lr', type=float, required=False)
args = parser.parse_args()


def train_encoder(index, state_dict, dataset, lr=1e-3, pre_iter=0, niters=10,
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

    model = MMmodels.Pretrain_Encoder()
    if state_dict:
        model.load_state_dict(state_dict)
    model = model.to(device).train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = losses.NTXentLoss(temperature=0.05)
    
    labels = [0]*9 + list(range(1,8))
    if index == 0:
        print(f'Labels: {labels}') # noqa
    
    loss = 100
    for it in range(pre_iter+1, pre_iter+niters+1):
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device) # noqa
        start = time.time()
        for batch_no, batch in enumerate(para_train_loader): # noqa
            print(f'Batch: {batch.size()}')
            images = batch
            print(f'Images: {images[0].size()}')
            image0 = [images[0]]*64
            print(f'Image0: {image0.size()}')
            '''
            image0 = MMdataset.mutations(image0)
            images = MMdataset.mutations(images)
            images = torch.stack([image0, images], dim=0)
            
            # Show the first batch of images
            if batch_no == 0 and index == 0:
                fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(12, 6))
                for i, ax in enumerate(axes.flatten()):
                    ax.imshow(images[i].permute(1, 2, 0))  # Assuming image tensor shape is (C, H, W)
                    ax.set_title(f"Image {i+1}")
                    ax.axis('off')
                plt.show()
            '''
            logits = model(images)
            train_loss = criterion(logits, labels)
            optimizer.zero_grad()
            train_loss.backward()
            xm.optimizer_step(optimizer)
            loss = train_loss.cpu()
        if index == 0:
            print("Master Process  |  Iter:{:4d}  |  Tr_loss: {:.4f}  |  Time: {}".format( # noqa
            it, loss.item(), MMutils.convert_seconds_to_time(time.time()-start))) # noqa

    if index == 0:
        MMutils.save_model(model.cpu(), current_dir, pre_iter+niters)


if __name__ == '__main__':
    print('Training Encoder...')
    model = MMmodels.Pretrain_Encoder()
    print(f'Total trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}') # noqa

    current_dir = os.path.dirname(os.path.realpath(__file__))

    pre_iter = 0
    state_dict = None
    if args.pretrain != 'no':
        pre_iter = int(args.pretrain)
        state_dict = torch.load(f'{current_dir}/model_iter_{pre_iter}.pth') # noqa
        print(f'Find model weights at {current_dir}/model_iter_{pre_iter}.pth, loading...') # noqa
        print(f'Now start to train from iter {pre_iter}...')

    gcs_path = 'gs://unlabelled-dataset/BreastMammography256/'
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
        current_dir     # current_dir
        ), start_method='forkserver')
