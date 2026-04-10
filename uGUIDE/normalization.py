import pickle
from typing import Any, Union
import torch


class BoundedNormalizer:

    def __init__(self, prior_min, prior_max, eps: float = 1e-6) -> None:

        self.prior_min = torch.as_tensor(prior_min, dtype=torch.float32)
        self.prior_max = torch.as_tensor(prior_max, dtype=torch.float32)

        # Avoid division by zero
        scale = self.prior_max - self.prior_min
        self.scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        self.eps = eps  # avoid exact 0/1 (important for sigmoid stability)

    def to(self, device: Union[str, torch.device]) -> "BoundedNormalizer":
        self.prior_min = self.prior_min.to(device)
        self.scale = self.scale.to(device)
        return self

    def __call__(self, data: Any) -> torch.Tensor:
        x = torch.as_tensor(data,
                            dtype=self.prior_min.dtype,
                            device=self.prior_min.device)

        # Step 1 — normalize to [0,1]
        x = (x - self.prior_min) / self.scale

        # Step 2 — clamp to avoid exact boundaries
        x = torch.clamp(x, self.eps, 1.0 - self.eps)

        return x

    def inverse(self, data: Any) -> torch.Tensor:
        x = torch.as_tensor(data,
                            dtype=self.prior_min.dtype,
                            device=self.prior_min.device)

        # Back to original range
        x = x * self.scale + self.prior_min

        return x


def get_bounded_normalizer(config: dict) -> BoundedNormalizer:

    prior_min = torch.tensor(
        [config['prior'][p][0] for p in config['prior'].keys()],
        dtype=torch.float32)

    prior_max = torch.tensor(
        [config['prior'][p][1] for p in config['prior'].keys()],
        dtype=torch.float32)

    return BoundedNormalizer(
        prior_min,
        prior_max,
    )


def save_bounded_normalizer(normalizer: BoundedNormalizer, path: str) -> None:
    data = {
        "prior_min":
        normalizer.prior_min.detach().cpu().tolist(),
        "prior_max":
        (normalizer.prior_min + normalizer.scale).detach().cpu().tolist(),
        "eps":
        normalizer.eps,
    }
    pickle.dump(data, open(path, "wb"))


def load_bounded_normalizer(path: str) -> BoundedNormalizer:
    data = pickle.load(open(path, "rb"))

    return BoundedNormalizer(
        prior_min=data["prior_min"],
        prior_max=data["prior_max"],
        eps=data.get("eps", 1e-6),
    )


class Normalizer:

    def __init__(self, loc, scale, use_log1p=False, clip_value=5.0) -> None:
        self.loc = torch.as_tensor(loc, dtype=torch.float32)
        scale_t = torch.as_tensor(scale, dtype=torch.float32)
        # avoid division by zero
        self.scale = torch.where(scale_t == 0, torch.ones_like(scale_t),
                                 scale_t)
        self.clip_value = clip_value
        self.use_log1p = use_log1p

    def to(self, device: Union[str, torch.device]) -> "Normalizer":
        self.loc = self.loc.to(device)
        self.scale = self.scale.to(device)
        return self

    def __call__(self, data: Any) -> torch.Tensor:
        x = torch.as_tensor(data, dtype=self.loc.dtype, device=self.loc.device)

        # Step 1 — log transform (if enabled)
        if self.use_log1p:
            x = torch.log1p(x)

        # Step 2 — robust normalization
        x = (x - self.loc) / self.scale

        # Step 3 — clipping
        x = torch.clamp(x, -self.clip_value, self.clip_value)

        return x

    def inverse(self, data):
        x = torch.as_tensor(data, dtype=self.loc.dtype, device=self.loc.device)

        # inverse of normalization
        x = x * self.scale + self.loc

        # inverse of log1p
        if self.use_log1p:
            x = torch.expm1(x)

        return x


def get_normalizer(data, use_log1p=False, clip_value=5.0) -> Normalizer:
    x = torch.as_tensor(data, dtype=torch.float32)

    if use_log1p:  # log(1 + x) to handle zero values and reduce skewness
        x = torch.log1p(x)

    median = x.median(0, keepdim=True).values
    q75 = torch.quantile(x, 0.75, dim=0, keepdim=True)
    q25 = torch.quantile(x, 0.25, dim=0, keepdim=True)
    iqr = q75 - q25

    # avoid division by zero
    iqr = torch.where(iqr == 0, torch.ones_like(iqr), iqr)

    return Normalizer(median, iqr, use_log1p=use_log1p, clip_value=clip_value)


def save_normalizer(normalizer: Normalizer, path: str) -> None:
    normalizer_weights = {
        "loc": normalizer.loc.detach().cpu().tolist(),
        "scale": normalizer.scale.detach().cpu().tolist(),
        "clip_value": normalizer.clip_value,
        "use_log1p": normalizer.use_log1p,
    }

    pickle.dump(normalizer_weights, open(path, "wb"))


def load_normalizer(path: str) -> Normalizer:
    normalizer_weights = pickle.load(open(path, "rb"))
    normalizer = Normalizer(
        loc=normalizer_weights["loc"],
        scale=normalizer_weights["scale"],
        clip_value=normalizer_weights.get("clip_value", 5.0),
        use_log1p=normalizer_weights.get("use_log1p", False))

    return normalizer
