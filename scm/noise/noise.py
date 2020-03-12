from abc import ABC, abstractmethod
from typing import Optional, Union, Collection


class Noise(ABC):
    """
    Abstract Base Class for the noise functions that can be provided to the SCM for noise generation
    when sampling.

    It currently demands only a ``__call__`` method to be implemented which can produce a vector output for the number of
    samples demanded, and a ``set_seed`` method for reseeding the noise for later reproducibility of results.
    """

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
