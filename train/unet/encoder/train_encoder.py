import os
import time
import argparse

import torch
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from Mammolibs import MMmodels, MMdataset, MMutils

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=str, required=True)
args = parser.parse_args()


def train_encoder(index, mmd, dataset, lr=1e-3, pre_iter=0, niters=100,
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
    model = MMmodels.Pretrain_Encoder().to(device).train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = 100
    for it in range(pre_iter+1, pre_iter+niters+1):
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device) # noqa
        start = time.time()
        for batch_no, batch in enumerate(para_train_loader): # noqa
            images1, images2 = batch
            logits1, logits2 = model(images1), model(images2)
            optimizer.zero_grad()
            train_loss = MMmodels.NT_Xent_loss(logits1, logits2)
            train_loss.backward()
            xm.optimizer_step(optimizer)
            loss = train_loss.cpu()
        print("Process: {:1d}  |  Iter:{:4d}  |  Tr_loss: {:.4f}  |  Time: {}".format( # noqa
        index, it, loss, MMutils.convert_seconds_to_time(time.time()-start))) # noqa

    MMutils.save_model(model.cpu(), save_path, niters)

    return model


if __name__ == '__main__':
    print('Training Encoder...')
    model = MMmodels.Pretrain_Encoder()

    current_dir = os.path.dirname(os.path.realpath(__file__))

    pre_iter = 0
    if args.pretrain == 'no':
        pass
    else:
        pre_iter = int(args.pretrain)
        state_dict = torch.load(f'{current_dir}/model_iter_{pre_iter}.pth') # noqa
        model.load_state_dict(state_dict)
        print(f'Find model weights at {current_dir}/model_iter_{pre_iter}.pth, loading...') # noqa

    print('Total trainable parameters = '
          f'{sum(p.numel() for p in model.parameters())}')

    gcs_path = 'gs://unlabelled-dataset/BreastMammography256/'
    dataset = MMdataset.MMImageSet(gcs_path)

    trained_model = xmp.spawn(train_encoder, args=(
        model,          # model
        dataset,        # dataset
        1e-3,           # lr
        pre_iter,       # pre_iter
        200,              # niters
        128,             # batch_size
        current_dir     # saving_dir
        ), start_method='forkserver')
