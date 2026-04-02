import torch
from torch import nn
import pyro.distributions.transforms as T


def build_nf(input_dim, nf_features=32, n_flows=5):

    transforms = []
    for t in range(n_flows):
        transform = T.conditional_affine_autoregressive(
            input_dim=input_dim,
            context_dim=nf_features,
            hidden_dims=[32, 32],
            log_scale_min_clip=-3.0,
            log_scale_max_clip=0.05,
            sigmoid_bias=2.0,
            stable=True,
        )
        transforms.append(transform)
        transforms.append(T.Permute(torch.randperm(input_dim)))

    transforms.append(T.SigmoidTransform())
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
