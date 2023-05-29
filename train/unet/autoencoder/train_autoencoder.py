import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from Mammolibs import MMmodels, MMdataset, MMutils

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=str, required=True)
parser.add_argument('--preiter', type=str, required=False)
parser.add_argument('--encoder', type=str, required=False)
parser.add_argument('--it', type=int, required=False)
parser.add_argument('--lr', type=float, required=False)
args = parser.parse_args()


def train_autoencoder(index, state_dict, dataset, lr=1e-3, pre_iter=0, niters=10, # noqa
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
    model = MMmodels.Autoencoder()
    if state_dict:
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device).train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss = 100
    for it in range(pre_iter+1, pre_iter+niters+1):
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device) # noqa
        start = time.time()
        for batch_no, batch in enumerate(para_train_loader): # noqa
            images = batch
            logits = model(images)
            optimizer.zero_grad()
            train_loss = criterion(logits, images)
            train_loss.backward()
            xm.optimizer_step(optimizer)
            loss = train_loss.cpu()
        print("Process: {:1d}  |  Iter:{:4d}  |  Tr_loss: {:.4f}  |  Time: {}".format( # noqa
        index, it, loss, MMutils.convert_seconds_to_time(time.time()-start))) # noqa

    if index == 0:
        MMutils.save_model(model.cpu(), save_path, niters)


if __name__ == '__main__':
    print('Training Autoencoder...')
    model = MMmodels.Autoencoder()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    if args.pretrain == 'no':
        pre_iter = int(args.encoder)
        state_dict = torch.load(f'{current_dir}/../encoder/model_iter_{pre_iter}.pth') # noqa
        state_dict = {k: v for k, v in state_dict.items() if 'encoder' in k} # noqa
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items()} # noqa
    else:
        pre_iter = int(args.pretrain)
        state_dict = torch.load(f'{current_dir}/model_iter_{pre_iter}.pth') # noqa
        print(f'Find model weights at {current_dir}/model_iter_{pre_iter}.pth, loading...') # noqa

    gcs_path = 'gs://unlabelled-dataset/BreastMammography256/'
    dataset = MMdataset.MMImageSet(gcs_path)

    n_iter = 20
    if args.it:
        n_iter = args.it

    lr = 3e-3
    if args.lr:
        lr = args.lr

    xmp.spawn(train_autoencoder, args=(
        state_dict,     # model
        dataset,        # dataset
        lr,             # lr
        pre_iter,       # pre_iter
        n_iter,         # niters
        128,            # batch_size
        current_dir     # current_dir
        ), start_method='forkserver')
