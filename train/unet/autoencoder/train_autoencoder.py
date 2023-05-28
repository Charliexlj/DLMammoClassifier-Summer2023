import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from Mammolibs import MMmodels, MMdataset, MMutils # noqa


def train_autoencoder(index, dataset, lr=1e-3, niters=1000,
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
    model = MMmodels.Mammo_Autoencoder().to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for it in range(1, niters+1):
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device) # noqa
        start = time.time()
        loss = 100000
        for batch_no, batch in enumerate(para_train_loader): # noqa
            if batch_no == 0:
                print('Start to train batch 1...')
            images1, images2 = batch
            logits1, logits2 = model(images1), model(images2)
            optimizer.zero_grad()
            train_loss = NT_Xent_loss(logits1, logits2)
            train_loss.backward()
            xm.optimizer_step(optimizer)
            loss = train_loss.cpu()
            if batch_no == 0 or batch_no == 9 or (batch_no+1) % 200 == 0:
                print(f'p{index} has completed {batch_no} batches in {MMutils.convert_seconds_to_time(time.time()-start)}') # noqa
        print("Process: {:1d}  |  Iter:{:4d}  |  Tr_loss: {:.4f}  |  Time: {}".format( # noqa
        index, it, loss, MMutils.convert_seconds_to_time(time.time()-start))) # noqa
    MMutils.save_model(model.cpu(), save_path, niters)
    return model


if __name__ == '__main__':
    print('Training Encoder...')
    model = Pretrain_Encoder()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    '''
    pretrained_weights = torch.load('./train/unet/encoder/model_epoch_200.pth')
    model.load_state_dict(pretrained_weights)
    '''
    print('Total trainable parameters = '
          f'{sum(p.numel() for p in model.parameters())}')

    gcs_path = 'gs://unlabelled-dataset/BreastMammography256/'
    dataset = MMdataset.MMImageSet(gcs_path)

    trained_model = xmp.spawn(train_encoder, args=(dataset, 1e-3, 10, 128, current_dir), start_method='fork') # noqa
