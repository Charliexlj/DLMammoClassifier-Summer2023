import torch

state_dict = torch.load('./train/unet/encoder/model_iter_15.pth') # noqa

for name in state_dict.keys():
    print(name)
