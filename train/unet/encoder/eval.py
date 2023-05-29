import os
import argparse

import torch
from torch.utils.data import DataLoader

from Mammolibs import MMmodels, MMdataset

parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()


def eval(model):
    model.eval()
    with torch.no_grad():
        test_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        images1, images2 = next(iter(test_dataloader))
        logits1, logits2 = model(images1), model(images2)
        test_loss = MMmodels.NT_Xent_loss(logits1, logits2)

        print('------------------------')
        print('Test Loss: ', test_loss)


if __name__ == '__main__':
    print('Testing Encoder...')
    model = MMmodels.Pretrain_Encoder()

    current_dir = os.path.dirname(os.path.realpath(__file__))

    niter = int(args.iter)
    state_dict = torch.load(f'{current_dir}/model_iter_{niter}.pth') # noqa
    model.load_state_dict(state_dict)
    print(f'Find model weights at {current_dir}/model_iter_{niter}.pth, loading...') # noqa

    gcs_path = 'gs://unlabelled-dataset/BreastMammography256/'
    dataset = MMdataset.MMImageSet(gcs_path)

    eval(model)
