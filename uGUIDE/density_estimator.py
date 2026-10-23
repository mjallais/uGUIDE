import torch
from torch import nn
import pyro.distributions.transforms as T
from pyro.distributions.conditional import ConditionalTransformModule


class ConstantConditionalTransformModule(ConditionalTransformModule):
    """
    Wrap an ordinary transform (e.g. Permute or Sigmoid) so that
    it can participate in a ConditionalTransformedDistribution.

    The transform itself does not depend on the context.

    For Permute transforms, the permutation tensor is explicitly
    moved to the same device as the conditioning context.
    """

    def __init__(self, transform):
        super().__init__()

        self.transform_type = type(transform).__name__

        if isinstance(transform, T.Permute):
            self.permutation = transform.permutation.clone().detach()
            self.dim = transform.dim

        else:
            self.transform = transform

    def condition(self, context):

        if self.transform_type == "Permute":
            permutation = self.permutation.to(context.device)

            return T.Permute(
                permutation=permutation,
                dim=self.dim,
            )

        return self.transform

    def clear_cache(self):
        if hasattr(self, "transform"):
            self.transform.clear_cache()


class ConditionalNF(nn.Module):
    """
    Container for the conditional normalizing-flow transforms.

    It is an nn.Module, so:
        nf.to(device)
        nf.parameters()

    both work.

    It is also iterable, so:
        ConditionalTransformedDistribution(base_dist, nf)

    works with Pyro.
    """

    def __init__(self, transforms):
        super().__init__()

        # Register all transforms as PyTorch submodules.
        self.transforms = nn.ModuleList(transforms)

    def __iter__(self):
        return iter(self.transforms)

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, index):
        return self.transforms[index]


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
        # transforms.append(T.Permute(torch.randperm(input_dim)))
        # Deterministic permutation
        if input_dim == 2:
            permutation = torch.tensor([1, 0])
        else:
            permutation = torch.arange(input_dim - 1, -1, -1)

        transforms.append(
            ConstantConditionalTransformModule(T.Permute(permutation)))

    # Bound every output to (0, 1).
    transforms.append(ConstantConditionalTransformModule(T.SigmoidTransform()))

    nf = ConditionalNF(transforms=transforms)

    return nf


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
