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
args = parser.parse_args()


class Pretrain_Encoder(nn.Module):
    def __init__(self, in_channel=1, out_vector=1024, num_filter=32):
        super(Pretrain_Encoder, self).__init__()
        self.encoder = MMmodels.Encoder(
            in_channel=in_channel, base_channel=num_filter)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(16*16*512, 2048),
            nn.ReLU()
            )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU()
            )
        self.fc3 = nn.Linear(2048, out_vector)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


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
    model = Pretrain_Encoder().to(device).train()

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
    loss = 100
    for it in range(pre_iter+1, pre_iter+niters+1):
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device) # noqa
        start = time.time()
        for batch_no, batch in enumerate(para_train_loader): # noqa
            images1, images2 = batch
            logits1, logits2 = model(images1), model(images2)
            optimizer.zero_grad()
            train_loss = NT_Xent_loss(logits1, logits2)
            train_loss.backward()
            xm.optimizer_step(optimizer)
            loss = train_loss.cpu()
        print("Process: {:1d}  |  Iter:{:4d}  |  Tr_loss: {:.4f}  |  Time: {}".format( # noqa
        index, it, loss, MMutils.convert_seconds_to_time(time.time()-start))) # noqa

    MMutils.save_model(model.cpu(), save_path, niters)

    return model


if __name__ == '__main__':
    print('Training Encoder...')
    model = Pretrain_Encoder()

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
        2,              # niters
        64,             # batch_size
        current_dir     # saving_dir
        ), start_method='forkserver')
