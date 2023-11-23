import torch
from torch import nn
import pyro.distributions.transforms as T

def build_nf(input_dim, nf_features=32):

    transforms = []
    for t in range(5):
        transform = T.conditional_affine_autoregressive(
            input_dim=input_dim,
            context_dim=nf_features,
            hidden_dims=[64, 64],
            log_scale_min_clip=-5.0,
            log_scale_max_clip=3.0,
            sigmoid_bias=2.0,
            stable=False,
        )
    transforms.append(transform)

    nf = T.ComposeTransformModule(parts=transforms)

    return nf


def get_nf(input_dim, nf_features, folder_path, nf_state_dict_file,
           load_state=False):

    nf = build_nf(input_dim=input_dim, nf_features=nf_features)
    
    if load_state:
        nf_state_dict = torch.load(folder_path / nf_state_dict_file)
        nf.load_state_dict(nf_state_dict)
        nf.eval()
    
    return nf
