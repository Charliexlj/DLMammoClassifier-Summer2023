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
parser.add_argument('--it', type=int, required=False)
parser.add_argument('--lr', type=float, required=False)
args = parser.parse_args()


def train_encoder(index, state_dict, dataset, lr=1e-3, pre_iter=0, niters=100,
                  batch_size=16):

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

    return model.cpu()


if __name__ == '__main__':
    print('Training Encoder...')

    current_dir = os.path.dirname(os.path.realpath(__file__))

    pre_iter = 0
    state_dict = None
    if args.pretrain != 'no':
        pre_iter = int(args.pretrain)
        state_dict = torch.load(f'{current_dir}/model_iter_{pre_iter}.pth') # noqa
        print(f'Find model weights at {current_dir}/model_iter_{pre_iter}.pth, loading...') # noqa
        print(f'Now start to train from iter {pre_iter}...')

    gcs_path = 'gs://unlabelled-dataset/BreastMammography256/'
    dataset = MMdataset.MMImageSet(gcs_path)

    n_iter = 20
    if args.it:
        n_iter = args.it

    lr = 3e-3
    if args.lr:
        lr = args.lr

    trained_model = xmp.spawn(train_encoder, args=(
        state_dict,     # model
        dataset,        # dataset
        lr,             # lr
        pre_iter,       # pre_iter
        n_iter,         # niters
        128,            # batch_size
        ), start_method='forkserver')

    MMutils.save_model(trained_model, current_dir, n_iter)
