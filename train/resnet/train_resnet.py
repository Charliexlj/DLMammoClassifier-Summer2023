import os
import time
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from Mammolibs import MMmodels, MMdataset, MMutils

parser = argparse.ArgumentParser()
parser.add_argument('--it', type=int, required=False)
parser.add_argument('--lr', type=float, required=False)
args = parser.parse_args()

load_model = 'utils/model_ResNet18.pt'

class MyModel(nn.Module):
    def __init__(self, models):
        super(MyModel, self).__init__()
        img_modules = list(models.children())[:-1]
        self.ModelA = nn.Sequential(*img_modules)
        self.relu = nn.ReLU()
        self.Linear1 = nn.Linear(512, 512, bias=True)  # additional fully connected layer
        self.Linear2 = nn.Linear(512, 1, bias=True)   # final layer for two class output

    def forward(self, x):
        x = self.ModelA(x)    # N x 512 x 1 x 1
        x = torch.flatten(x, 1)
        x = self.relu(self.Linear1(x))  # add relu activation function after first fc layer
        x = self.Linear2(x)   # final output layer
        return x

if os.path.exists(load_model):
    checkpoint=torch.load(load_model,map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state'],strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state'],strict=False)
    print("model loaded successfully")
    print('starting training after epoch: ',checkpoint['epoch'])

def train_resnet(index, state_dict, dataset, lr=1e-3, pre_iter=0, niters=10, # noqa
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
    models = torchvision.models.resnet18(pretrained=True)
    model = MyModel(models)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if os.path.exists(load_model):
        checkpoint=torch.load(load_model,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state'],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state'],strict=False)
        print("model loaded successfully")
        print('starting training after epoch: ',checkpoint['epoch'])
    model = model.to(device).train()

    criterion = nn.BCEWithLogitsLoss()
    loss = 100
    for it in range(pre_iter+1, pre_iter+niters+1):
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device) # noqa
        start = time.time()
        for batch_no, batch in enumerate(para_train_loader): # noqa
            patches, labels = batch
            labels = labels.unsqueeze(1)
            print(labels[0])
            logits = model(patches)
            optimizer.zero_grad()
            train_loss = criterion(logits, labels)
            train_loss.backward()
            xm.optimizer_step(optimizer)
            loss = train_loss.cpu()
        if index == 0:
            print("Master Process  |  Iter:{:4d}  |  Tr_loss: {:.4f}  |  Time: {}".format( # noqa
            it, loss, MMutils.convert_seconds_to_time(time.time()-start))) # noqa

    if index == 0:
        MMutils.save_model(model.cpu(), save_path, niters+pre_iter)


if __name__ == '__main__':
    print('Training Resnet...')
    '''
    model = MMmodels.Autoencoder()
    print(f'Total trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}') # noqa
    '''
    # print(f'Total trainable parameters = {6963697}') # noqa

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
        load_model,     # model
        dataset,        # dataset
        lr,             # lr
        0,       # pre_iter
        n_iter,         # niters
        64,            # batch_size
        current_dir     # current_dir
        ), start_method='forkserver')
