from abc import ABC, abstractmethod
from typing import Optional, Union, Collection, Dict


class Noise(ABC):
    """
    Abstract Base Class for the noise functions that can be provided to the SCM for noise generation
    when sampling.

    It currently demands only a ``__call__`` method to be implemented which can produce a vector output for the number of
    samples demanded, and a ``set_seed`` method for reseeding the noise for later reproducibility of results.
    """

    def __init__(self):
        self.params: Dict[str, Union[float, int]] = dict()
        self._noise_description = ""

    @property
    def noise_description(self):
        return self.noise_description

    @noise_description.setter
    def noise_description(self, desc: str):
        desc_str = f"{' '.join(map(str.capitalize, desc.split('_')))}"
        if self.params:
            desc_str += "(" + ", ".join(f"{param}={round(value, 2)}" for param, value in self.params.items())
        self._noise_description = desc_str

    @abstractmethod
    def set_seed(self, seed):
        """
        Seed the noise with the given value.
        Parameters
        ----------
        seed: int,
            The random seed for random number generation.
        """

    @abstractmethod
    def __call__(self, size, **kwargs):
        """
        Abstract method for the call on the noise object.

        Parameters
        ----------
        size: int,
            The number of samples that should be generated.
        kwargs: dict,
            Any keyword arguments that a future implementation might need.

        Returns
        -------
        Collection of samples (preferably a numpy array).
        """

    def __str__(self):
        return self._noise_description
