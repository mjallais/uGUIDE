import torch
from torch import nn
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.distributions.conditional import ConstantConditionalTransform, ConditionalTransform


class FlowModule(nn.Module, ConditionalTransform):
    """Wraps Pyro transforms while registering parameters and forwarding context."""

    def __init__(self, cond_transforms, ar_modules):
        super().__init__()
        self.ar_modules = nn.ModuleList(ar_modules)
        self.cond_transforms = cond_transforms

    def condition(self, context):
        """Pass context to conditional transforms, return composite transform."""
        conditioned = [
            t.condition(context) if isinstance(t, ConditionalTransform) else t
            for t in self.cond_transforms
        ]
        return T.ComposeTransform(conditioned)

    def _apply(self, fn):
        super()._apply(fn)
        for ct in self.cond_transforms:
            t = getattr(ct, "transform", getattr(ct, "cache", ct))
            for attr in ["permutation", "inv_permutation"]:
                if hasattr(t, attr):
                    tensor = getattr(t, attr)
                    if isinstance(tensor, torch.Tensor):
                        setattr(t, attr, fn(tensor))
        return self

    def __iter__(self):
        return iter(self.cond_transforms)

    def __len__(self):
        return len(self.cond_transforms)


def build_nf(input_dim, nf_features=32, n_flows=5):
    cond_transforms = []
    ar_modules = []

    for _ in range(n_flows):
        # 1. Conditional Autoregressive Layer
        ar = T.conditional_affine_autoregressive(
            input_dim=input_dim,
            context_dim=nf_features,
            hidden_dims=[32, 32],
            log_scale_min_clip=-5.0,
            log_scale_max_clip=2.0,
            sigmoid_bias=2.0,
            stable=True,
        )
        ar_modules.append(ar)
        cond_transforms.append(ar)

        # 2. Static Permute wrapped as ConstantConditionalTransform
        perm = T.Permute(torch.randperm(input_dim))
        cond_transforms.append(ConstantConditionalTransform(perm))

    # 3. Static Sigmoid wrapped as ConstantConditionalTransform
    sig = T.SigmoidTransform()
    cond_transforms.append(ConstantConditionalTransform(sig))

    return FlowModule(cond_transforms, ar_modules)


# def build_nf(input_dim, nf_features=32, n_flows=5):

#     transforms = []
#     for _ in range(n_flows):
#         # 1. Conditional Affine Autoregressive Transform
#         transform = T.conditional_affine_autoregressive(
#             input_dim=input_dim,
#             context_dim=nf_features,
#             hidden_dims=[32, 32],
#             log_scale_min_clip=-3.0,
#             log_scale_max_clip=0.05,
#             sigmoid_bias=2.0,
#             stable=True,
#         )
#         transforms.append(transform)

#         # 2. Fixed permutation transform
#         transforms.append(T.TransformModule(T.permute(input_dim=input_dim)))
#         # transforms.append(T.Permute(torch.randperm(input_dim)))
#         # # Deterministic permutation
#         # permutation = torch.roll(
#         #     torch.arange(input_dim),
#         #     shifts=t + 1,
#         # )
#         # transforms.append(ConditionalPermute(permutation))

#     # 3. Final Sigmoid transform to map output to (0, 1)
#     # transforms.append(T.SigmoidTransform())
#     transforms.append(T.TransformModule(T.SigmoidTransform()))

#     # Compose module registers parameters for Pyro/torch.nn
#     nf = T.ComposeTransformModule(transforms)
#     # nf = ConditionalComposeTransformModule(transforms)

#     return nf


def get_nf(input_dim,
           nf_features,
           n_flows=5,
           pretrained_state=None,
           device='cpu'):

    nf = build_nf(input_dim=input_dim,
                  nf_features=nf_features,
                  n_flows=n_flows)

    if pretrained_state is not None:
        nf_state_dict = torch.load(pretrained_state,
                                   map_location=torch.device(device))
        nf.load_state_dict(nf_state_dict)
        nf.eval()

    return nf
