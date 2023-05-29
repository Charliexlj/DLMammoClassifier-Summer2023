import torch
from Mammolibs import MMmodels

en_state_dict = torch.load('./train/unet/encoder/model_iter_15.pth') # noqa

model = MMmodels.Autoencoder()
original_params = {k: v.clone() for k, v in model.state_dict().items()}
model.load_state_dict(en_state_dict, strict=False)

updated_params = []
for k, v in model.state_dict().items():
    if k in original_params and not torch.allclose(original_params[k], v):
        updated_params.append(k)

print('Updated layers:')
for name in updated_params:
    print(name)
