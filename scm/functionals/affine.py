from typing import Optional, Dict, Union, Collection, Hashable

from .functionals import Functional

import numpy as np


class AffineFunctional(Functional):
    r"""
    The affine Functional function of the form:
        f(X_S, N) = offset + \sum_{i \in S} a_i * X_i + noise_coeff * N
    """

    def __init__(
        self,
        offset: float = 0.0,
        *coefficients,
        vars: Optional[Collection[Hashable]] = None,
        **base_kwargs
    ):
        self.offset = offset
        self.coefficients = np.asarray(coefficients)

        if vars is None:
            vars = tuple(str(i) for i in range(len(coefficients)))
        base_kwargs.update(dict(vars=vars))
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, *args, **kwargs):
        return self.offset + self.coefficients @ args

    def __len__(self):
        return 1 + len(self.args_to_position)

    def function_str(self, variable_names=None):
        rep = ""
        if self.offset != 0:
            rep += str(round(self.offset, 2))
        for i, c in enumerate(self.coefficients):
            if c != 0:
                rep += f" + {round(c, 2)} {variable_names[i + 1]}"
        return rep

    # @staticmethod
    # def random_factory(nr_variables, seed=None):
    #     rs = np.random.default_rng(seed)
    #     offset = rs.normal()
    #     coeffs = rs.normal(loc=0, scale=1, size=nr_variables)
    #     return LinearFunctional(1, offset, *coeffs)
