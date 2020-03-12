from .functionals import Functional, Functional

from scipy.special import expit
from typing import Callable, Optional, Collection, Hashable
import types
import numpy as np


class Sigmoid(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg, **kwargs):
        return expit(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"sigmoid({list(self.get_arg_names())[0]})"
