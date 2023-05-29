import torch
from Mammolibs import MMmodels

en_state_dict = torch.load('./train/unet/encoder/model_iter_15.pth') # noqa
model = MMmodels.Autoencoder()

for name in en_state_dict.keys():
    print(name)

for name in model.state_dict.keys():
    print(name)
