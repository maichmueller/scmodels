from .functionals import Functional

import numpy as np
from numpy.polynomial import polynomial
from typing import List, Collection, Union, Optional


class Polynomial(Functional):
    r"""
    A polynomial functional function of the form:

    .. math:: f(X_S, N) = offset + \sum_{i \in S} {\sum^{p_i}}_{k = 0} c_{ik} * X_i^k

    """
    def __init__(self, *coefficients_list: Collection[Union[int, float]], **base_kwargs):
        polynomials = []
        if len(coefficients_list) > 0:
            for coefficients in coefficients_list:
                polynomials.append(polynomial.Polynomial(coefficients))
        self.polynomials = polynomials
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, *args, **kwargs):
        return sum((poly(arg) for poly, arg in zip(self.polynomials, args)))

    def __len__(self):
        return len(self.get_arg_names())

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        functional = []
        for poly, var in zip(self.polynomials, variable_names):
            coeffs = poly.coef
            this_poly = []
            for deg, c in enumerate(coeffs):
                if c != 0:
                    this_poly.append(
                        f"{round(c, 2)} {var}{f'**{deg}' if deg != 1 else ''}"
                    )
            functional.append(" + ".join(this_poly))
        return " + ".join(functional)

    @staticmethod
    def random_factory(nr_variables, seed=None):
        rs = np.random.default_rng(seed)
        coeffs = []
        for n in range(nr_variables + 1):
            deg = rs.integers(1, 6)  # allows degrees d <= 5
            coeffs.append(np.random.normal(loc=0, scale=0.1, size=deg))
        return PolynomialFunctional(*coeffs)
