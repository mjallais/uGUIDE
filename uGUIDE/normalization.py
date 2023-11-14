import pickle
import numpy as np


class Normalizer:
    def __init__(self, loc, scale) -> None:
        self.loc = loc
        self.scale = np.where(scale == 0, 1., scale)

    def __call__(self, data) -> np.ndarray:
        return (data - self.loc) / self.scale

    def inverse(self, data):
        return data * self.scale + self.loc


def get_normalizer(data: np.ndarray) -> Normalizer:
    mean = data.mean(0, keepdims=True).astype(np.float32)
    std = data.std(0, keepdims=True).astype(np.float32)

    return Normalizer(mean, std)


def save_normalizer(normalizer: Normalizer, path: str) -> None:
    normalizer_weights = {
        "loc": normalizer.loc,
        "scale": normalizer.scale,
    }

    pickle.dump(normalizer_weights, open(path, "wb"))


def load_normalizer(path: str) -> Normalizer:
    normalizer_weights = pickle.load(open(path, "rb"))
    normalizer = Normalizer(
        loc=normalizer_weights["loc"], scale=normalizer_weights["scale"]
    )

    return normalizer
