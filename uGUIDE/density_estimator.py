import torch
from torch import nn
import pyro.distributions.transforms as T


def build_nf(input_dim, nf_features=32, n_flows=5, flow_type='MAF'):

    transforms = []
    for t in range(n_flows):
        if flow_type == 'MAF':
            transform = T.conditional_affine_autoregressive(
                input_dim=input_dim,
                context_dim=nf_features,
                hidden_dims=[64, 64],
                log_scale_min_clip=-3.0,
                log_scale_max_clip=1.5,
                sigmoid_bias=2.0,
                stable=True,
            )
        elif flow_type == 'NSF':
            transforms.append(
                T.conditional_spline_autoregressive(
                    input_dim=input_dim,
                    context_dim=nf_features,
                    hidden_dims=[64, 64],
                    count_bins=10,  # number of spline segments
                    bound=8.0,
                ))
        transforms.append(transform)
        transforms.append(T.Permute(torch.randperm(input_dim)))

    nf = T.ComposeTransformModule(parts=transforms)

    return nf


def get_nf(input_dim, nf_features, n_flows=5, pretrained_state=None):

    nf = build_nf(input_dim=input_dim,
                  nf_features=nf_features,
                  n_flows=n_flows)

    if pretrained_state is not None:
        nf_state_dict = torch.load(pretrained_state)
        nf.load_state_dict(nf_state_dict)
        nf.eval()

    return nf
