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
        self._permutation_buffers = {}
        for index, transform in enumerate(cond_transforms):
            if isinstance(transform, ConstantConditionalTransform):
                transform = transform.transform
            if isinstance(transform, T.Permute):
                name = f"flow_permutation_{index}"
                self.register_buffer(name, transform.permutation.clone())
                self._permutation_buffers[index] = name

    def condition(self, context):
        """Pass context to conditional transforms, return composite transform."""
        conditioned = [
            t.condition(context) if isinstance(t, ConditionalTransform) else t
            for t in self
        ]
        return T.ComposeTransform(conditioned)

    def __iter__(self):
        for index, transform in enumerate(self.cond_transforms):
            if index in self._permutation_buffers:
                # Rebuild from the saved buffer so loading and device moves also
                # update the transform, including its cached inverse permutation.
                transform = ConstantConditionalTransform(
                    T.Permute(getattr(self, self._permutation_buffers[index])))
            yield transform

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
            # Allow expansion as well as contraction in logit space. The stable
            # variant can saturate at scale=1 and suppress its conditional mean.
            stable=False,
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
        if any(name not in nf_state_dict
               for name in nf._permutation_buffers.values()):
            raise ValueError(
                "This flow checkpoint is missing the inter-layer permutations. "
                "Retrain with load_state=False to save a complete flow; the "
                "original permutations cannot be recovered from this checkpoint."
            )
        nf.load_state_dict(nf_state_dict)
        nf.eval()

    return nf
