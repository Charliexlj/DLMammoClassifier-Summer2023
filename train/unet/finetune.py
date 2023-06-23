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

from Mammolibs import MMmodels, MMdataset, MMutils

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=str, required=True)
parser.add_argument('--it', type=int, required=False)
parser.add_argument('--lr', type=float, required=False)
parser.add_argument('--autoencoder', type=str, required=False)
args = parser.parse_args()


class mIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        N = inputs.size()[0]
        inputs = nn.functional.softmax(inputs, dim=1)
        inter = inputs * target_oneHot
        inter = inter.view(N, self.classes, -1).sum(2)
        union = inputs + target_oneHot - (inputs*target_oneHot)
        union = union.view(N, self.classes, -1).sum(2)
        loss = inter/union
        return 1-loss.mean()


def finetune(index, state_dict, dataset, lr=1e-3, pre_iter=0, niters=10,
             batch_size=16, current_dir='/home'):

    device = xm.xla_device()

    model = MMmodels.UNet()
    if state_dict:
        unet_state_dict = model.state_dict()
        partial_state_dict = {k: v for k, v in state_dict.items() if k in unet_state_dict and v.size() == unet_state_dict[k].size()} # noqa
        model.load_state_dict(partial_state_dict, strict=False)
    model = model.to(device).train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.BCEWithLogitsLoss()
    # jaccard = JaccardIndex(num_classes=2, task='binary')
    criterion = mIoULoss()
    
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
        for batch_no, batch in enumerate(para_train_loader): # noqa
            images, labels = batch
            '''
            image_labels = torch.stack((images, labels), dim=1)
            image_labels = torch.stack([MMdataset.mutations(image_label) for image_label in image_labels]) # noqa
            images = image_labels[:, 0, :, :, :]
            labels = image_labels[:, 1, :, :, :]
            '''
            logits = model(images)
            train_loss = criterion(logits, labels)
            optimizer.zero_grad()
            train_loss.backward()
            xm.optimizer_step(optimizer)
            loss = train_loss.cpu()
            
            if index == 0 and batch_no == 0 and it % 3 == 1:
                np_roi = labels.cpu().numpy()[:4]  # [:4].numpy().reshape((4, 2, 256, 256))
                print("np_roi: ", np_roi.shape)
                
                logits_np = logits.cpu().detach().numpy()[:4]  # [:4]
                print("logits shape: ", logits_np.shape)
                
                image_np = images.cpu().numpy()[:4].reshape((4, 256, 256))
                roi_np = labels.cpu().numpy()[:4].reshape((4, 2, 256, 256))

                fig, axs = plt.subplots(5, 4, figsize=(12, 15))
                for i in range(4):
                    # plot image
                    axs[0, i].imshow(image_np[i], cmap='gray')
                    axs[0, i].set_title(f'Image {i+1}')
                    axs[0, i].axis('off')
                    
                    axs[1, i].imshow(roi_np[i][0], cmap='gray')
                    axs[1, i].set_title(f'Label[0] {i+1}')
                    axs[1, i].axis('off')
                    
                    axs[2, i].imshow(roi_np[i][1], cmap='gray')
                    axs[2, i].set_title(f'Label[1] {i+1}')
                    axs[2, i].axis('off')
                    
                    axs[3, i].imshow(logits_np[i][0], cmap='gray')
                    axs[3, i].set_title(f'Logit[0] {i+1}')
                    axs[3, i].axis('off')
                    
                    axs[4, i].imshow(logits_np[i][1], cmap='gray')
                    axs[4, i].set_title(f'Logit[1] {i+1}')
                    axs[4, i].axis('off')

                plt.tight_layout()
                plt.savefig(f'plot_{it}.png')
                print(f'saved plot_{it}.png')
            
            if index == 0 and batch_no % 10 == 0:
                print("Batch:{:4d}  |  Iter:{:4d}  |  Tr_loss: {:.4f}".format( # noqa
                batch_no, it, loss)) # noqa
        if index == 0:
            print("=======================================================================") # noqa
            print("Master Process  |  Iter:{:4d}  |  Tr_loss: {:.4f}  |  Time: {}".format( # noqa
            it, loss, MMutils.convert_seconds_to_time(time.time()-start))) # noqa
            print("=======================================================================") # noqa
    if index == 0:
        MMutils.save_model(model.cpu(), current_dir, pre_iter+niters)
        image_test, label_test = images.cpu().numpy(), labels.cpu().numpy()
        logits_np = logits.cpu().detach().numpy()
        print("logits shape: ", logits_np.shape)
        fig, axs = plt.subplots(3, 4, figsize=(12, 16))
        for i in range(4):
            # plot image
            axs[0, i].imshow(image_test[i][0], cmap='gray')
            axs[0, i].set_title(f'Image {i+1}')
            axs[0, i].axis('off')
            
            axs[1, i].imshow(label_test[i][0], cmap='gray')
            axs[1, i].set_title(f'Label[0] {i+1}')
            axs[1, i].axis('off')
            
            axs[2, i].imshow(logits_np[i][0], cmap='gray')
            axs[2, i].set_title(f'Logit[0] {i+1}')
            axs[2, i].axis('off')

        plt.tight_layout()
        plt.savefig(f'plot_test_{it}.png')
        print(f'saved plot_test_{it}.png')


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

    gcs_path = 'gs://last_dataset/labelled-dataset/BreastMammography/Benign/'
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
        128,            # batch_size
        current_dir     # current_dir
        ), start_method='forkserver')
    