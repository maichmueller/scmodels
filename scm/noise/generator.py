from . import Noise
from typing import *

import numpy as np
from numpy.random import Generator, PCG64
from functools import partial, wraps


class NoiseGenerator(Noise):
    """
    A simple feed forward convenience class to generate different numpy provided distributions to the user.

    For numpy distributions the list is:

    -- Beta distribution
        - beta(a, b[, size])
    -- Binomial distribution
        - binomial(n, p[, size])
    -- chi-square distribution
        - chisquare(df[, size])
    -- Dirichlet distribution
        - dirichlet(alpha[, size])
    -- exponential distribution
        - exponential([scale, size])
    -- F distribution
        - f(dfnum, dfden[, size])
    -- Gamma distribution
        - gamma(shape[, scale, size])
    -- geometric distribution
        - geometric(p[, size])
    -- Gumbel distribution
        - gumbel([loc, scale, size])
    -- Hypergeometric distribution
        - hypergeometric(ngood, nbad, nsample[, size])
    -- Laplace or double exponential distribution with specified location (or mean) and scale (decay)
        - laplace([loc, scale, size])
    -- logistic distribution
        - logistic([loc, scale, size])
    -- log-normal distribution
        - lognormal([mean, sigma, size])
    -- logarithmic series distribution
        - logseries(p[, size])
    -- multinomial distribution
        - multinomial(n, pvals[, size])
    -- multivariate normal distribution
        - multivariate_normal(mean, cov[, size, …])
    -- negative binomial distribution
        - negative_binomial(n, p[, size])
    -- noncentral chi-square distribution
        - noncentral_chisquare(df, nonc[, size])
    -- noncentral F distribution
        - noncentral_f(dfnum, dfden, nonc[, size])
    -- normal (Gaussian) distribution
        - normal([loc, scale, size])
    -- Pareto II or Lomax distribution with specified shape
        - pareto(a[, size])
    -- Poisson distribution
        - poisson([lam, size])
    -- [0, 1] from a power distribution with positive exponent a - 1
        - power(a[, size])
    -- Rayleigh distribution
        - rayleigh([scale, size])
    -- standard Cauchy distribution with mode = 0
        - standard_cauchy([size])
    -- standard exponential distribution
        - standard_exponential([size, dtype, method, out])
    -- standard Gamma distribution
        - standard_gamma(shape[, size, dtype, out])
    -- standard Normal distribution (mean=0, stdev=1)
        - standard_normal([size, dtype, out])
    -- standard Student’s t distribution with df degrees of freedom
        - standard_t(df[, size])
    -- triangular distribution over the interval [left, right]
        - triangular(left, mode, right[, size])
    -- uniform distribution
        - uniform([low, high, size])
    -- von Mises distribution
        - vonmises(mu, kappa[, size])
    -- Wald, or inverse Gaussian, distribution
        - wald(mean, scale[, size])
    -- Weibull distribution
        - weibull(a[, size])
    -- Zipf distribution
        - zipf(a[, size])

    For Scipy distributions please refer to:
    https://docs.scipy.org/doc/scipy/reference/stats.html
    """
    standard_distr_names = {
        "standard_cauchy": "Cauchy"
    }

    def __init__(self, distribution_str: Optional[str] = None, source="numpy", seed=None, **distribution_params):
        self.params = distribution_params
        self.distribution_str = distribution_str
        self.source = source
        self.seed = seed

        self.distribution = None
        self.set_distribution(distribution_str, source=source, seed=seed)

    def set_source(self, source):
        self.source = source

    def set_seed(self, seed: int):
        self.set_distribution(self.distribution_str, self.source, seed=seed)

    def _apply_dist_params(func: Callable):
        @wraps(func)
        def dist(self, *args, **kwargs):
            return func(self, *args, **kwargs, **self.params)
        return dist

    def set_distribution(self, distribution_str, source, seed=None):
        seed = self.seed if seed is None else seed
        if distribution_str is None:
            self.distribution = lambda size: np.zeros(size)
            return
        if source == "numpy":
            try:
                rng = np.random.default_rng(seed)
                self.distribution = eval(f"rng.{distribution_str}")
            except AttributeError as a:
                raise ValueError(
                    f"No distribution found in source numpy for "
                    f"distribution={distribution_str}."
                )
        elif source == "scipy":
            try:
                exec(f"from scipy.stats import {distribution_str}")
                distribution = eval(f"{distribution_str}")
                distribution.random_state = seed
                self.distribution = distribution.rvs
            except ImportError as i:
                raise ValueError(
                    f"No distribution found in source scipy for "
                    f"distribution={distribution_str}."
                )
        else:
            raise ValueError(
                f"Source '{source}' unknown/not supported yet."
            )

    @_apply_dist_params
    def __call__(self, size=1, **kwargs):
        return self.distribution(size=size, **kwargs)

    _apply_dist_params = staticmethod(_apply_dist_params)

    def __str__(self):
        s = f"{' '.join(map(str.capitalize, self.distribution_str.split('_')))}"
        if self.params:
            s += "("
            for param, value in self.params.items():
                s += f"{param}={round(value, 2)}, "

            s = s[0:-2] + ")"  # remove last comma and whitespace if there were params
        return s
