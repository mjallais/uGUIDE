import torch
from torch import nn
import pyro.distributions.transforms as T

def build_nf():
    input_dim = SIZE_THETA
    context_dim = 32

    transforms = []
    for t in range(5):
        transform = T.conditional_affine_autoregressive(
            input_dim=input_dim,
            context_dim=context_dim,
            hidden_dims=[64, 64],
            log_scale_min_clip=-5.0,
            log_scale_max_clip=3.0,
            sigmoid_bias=2.0,
            stable=False,
        )
    transforms.append(transform)

    nf = T.ComposeTransformModule(parts=transforms)

    return nf