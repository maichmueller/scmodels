from .noise import Noise
from typing import Optional, Union, Collection
import numpy as np


class DiscreteNoise(Noise):
    def __init__(
        self,
        values: Collection[Union[float, int]],
        probabilities: Optional[Collection[float]] = None,
        seed: Optional[int] = None,
    ):
        self.values = np.array(values)
        self.probabilities = (
            np.array(probabilities)
            if probabilities is not None
            else np.ones_like(self.values) / len(self.values)
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, size, **kwargs):
        return self.rng.choice(a=self.values, p=self.probabilities, size=size)

    def __str__(self):
        s = (
            f"Discrete("
            f"v={np.array2string(self.values.round(2), threshold=5)}, "
            f"p={np.array2string(self.probabilities.round(2), threshold=5)}"
            f")"
        )
        return s
