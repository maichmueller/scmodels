from scm.functionals import *
from scm.scm import SCM
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
from test.builld_scm_funcs import *


def manual_standard_sample(n, noise_func, dtype, names, seed):
    sample = np.empty((n, 5), dtype=dtype)
    sample[:, 0] = noise_func(n, seed)
    sample[:, 1] = 1 + noise_func(n, seed+1) + 2 * sample[:, 0]
    sample[:, 2] = 1 + noise_func(n, seed+2) + 3 * sample[:, 0] + 2 * sample[:, 1]
    sample[:, 3] = (
        noise_func(n, seed+3)
        + Polynomial([0, 1, 0.5])(sample[:, 1])
        + Polynomial([0, 0, 4])(sample[:, 2])
    )
    sample[:, 4] = (
        noise_func(n, seed+4)
        + Polynomial([0, 0, 1.5])(sample[:, 0])
        + Polynomial([0, 2])(sample[:, 2])
    )
    sample = pd.DataFrame(sample, columns=names)
    return sample


def noise(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def test_scm_build():
    cn = build_scm_linandpoly()
    nodes_in_graph = list(cn.graph.nodes)
    assert nodes_in_graph == ["X_0", "X_1", "X_2", "X_3", "Y"]


def test_scm_sample():
    cn = build_scm_linandpoly()
    scm_sample = cn.sample(10)
    sample = manual_standard_sample(
        10, noise, scm_sample.values.dtype, list(cn.graph.nodes), 0
    )
    sample_order_scm = list(cn.get_variables())
    sample = sample[sample_order_scm]
    # floating point inaccuracy needs to be accounted for
    assert (sample - scm_sample).abs().values.sum() < 1e-10


def test_scm_intervention():
    cn = build_scm_linandpoly()

    # do the intervention
    cn.intervention(
        {
            "X_3": {
                "parents": ["X_0", "Y"],
                "functional": Affine(1, 0, 3.3, 3.3),
                "noise": NoiseGenerator("t", df=1, source="scipy", seed=0),
            }
        }
    )
    n = 10

    scm_sample_interv = cn.sample(n)
    sample = np.empty((n, 5), dtype=scm_sample_interv.values.dtype)
    sample[:, 0] = noise(n, 0)
    sample[:, 1] = 1 + noise(n, 1) + 2 * sample[:, 0]
    sample[:, 2] = 1 + noise(n, 2) + 3 * sample[:, 0] + 2 * sample[:, 1]
    sample[:, 4] = (
        noise(n, 4)
        + Polynomial([0, 0, 1.5])(sample[:, 0])
        + Polynomial([0, 2])(sample[:, 2])
    )
    sample[:, 3] = NoiseGenerator("t", df=1, source="scipy", seed=0)(n) + 3.3 * (
        sample[:, 0] + sample[:, 4]
    )
    sample = pd.DataFrame(sample, columns=list(cn.graph.nodes))
    sample_in_scm_order = sample[list(cn.get_variables())]

    assert (sample_in_scm_order - scm_sample_interv).abs().values.sum() < 1e-10

    # from here on the cn should work as normal again
    cn.undo_intervention()

    # reseeding needs to happen as the state of the initial noise distributions is further advanced than
    # a newly seeded noise by noise()
    cn.reseed(0)

    man_sample = manual_standard_sample(
        n, noise, scm_sample_interv.values.dtype, list(cn.graph.nodes), 0
    )[list(cn.get_variables())]
    new_cn_sample = cn.sample(n)
    assert (man_sample - new_cn_sample).abs().values.sum() < 1e-10
