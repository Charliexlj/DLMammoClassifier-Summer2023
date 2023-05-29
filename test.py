import torch

state_dict = torch.load('./train/unet/encoder/model_iter_15.pth') # noqa
print(state_dict)
