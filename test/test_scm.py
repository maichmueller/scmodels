import random
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
from test.build_scm import *
from scm.parser import parse_assignments


def manual_standard_sample(n, dtype, names, seed):
    rng = random.Random(seed)
    rng.normalvariate(0, 1)
    noise_func = lambda size: np.array(
        [rng.normalvariate(mu=0, sigma=1) for i in range(size)]
    )
    sample = np.empty((n, 5), dtype=dtype)
    sample[:, 0] = noise_func(n)
    sample[:, 1] = 1 + noise_func(n) + 2 * sample[:, 0] ** 2
    sample[:, 2] = 1 + noise_func(n) + 3 * sample[:, 0] + 2 * sample[:, 1]
    sample[:, 3] = (
            noise_func(n)
            + sample[:, 1]
            + 0.5 * np.sqrt(np.abs(sample[:, 1]))
            + 4 * np.log(np.abs(sample[:, 2]))
    )
    sample[:, 4] = (
            noise_func(n) + Polynomial([0, 0, 0.05])(sample[:, 0]) + 2 * sample[:, 2]
    )
    sample = pd.DataFrame(sample, columns=names)
    return sample


def test_parsing():
    func_map = parse_assignments(["Z_1 = Noise + 2*log(Y), Noise ~ Normal(0,1)"])
    assert list(func_map.keys()) == ["Z_1"]
    assert func_map["Z_1"][0] == ["Y"]
    assert func_map["Z_1"][1] == "Noise + 2*log(Y)"
    assert str(func_map["Z_1"][2]) == "Noise"

    func_map = parse_assignments(["X = N + sqrt(X_45 ** M + 342 * 2) / (  43 * FG_2) + P, N ~ Normal(0,1)"])
    assert list(func_map.keys()) == ["X"]
    assert func_map["X"][0] == ["X_45", "M", "FG_2", "P"]
    assert func_map["X"][1] == "N + sqrt(X_45 ** M + 342 * 2) / (  43 * FG_2) + P"
    assert str(func_map["X"][2]) == "N"


def test_scm_from_strings():
    scm = SCM(
        [
            "X = N, N ~ Normal(0,1)",
            "Y_0 = M + 2 * exp(X), M ~ StudentT(2)",
            "Y_1 = M + 2 * exp(sqrt(X)), M ~ Normal(0, 0.1)",
            "Z = P * sqrt(Y_0), P ~ Exponential(5.3)",
        ]
    )
    assert np.isin(scm.get_variables(), ["X", "Y_0", "Y_1", "Z"]).all()
    

def test_scm_build():
    cn = build_scm_simple()
    nodes_in_graph = sorted(cn.dag.nodes)
    assert nodes_in_graph == sorted(["X_0", "X_1", "X_2", "X_3", "X_4", "X_5", "Y"])


def test_scm_sample_partial():
    cn = build_scm_linandpoly(seed=0)
    n = 1000
    random.seed(0)
    sample_vars = cn.sample(n, ["X_0"]).columns
    assert sample_vars[0] == "X_0" and len(sample_vars) == 1

    sample_vars = cn.sample(n, ["X_2"]).columns
    assert np.isin(["X_0", "X_1", "X_2"], sample_vars).all() and len(sample_vars) == 3

    sample_vars = cn.sample(n, ["Y"]).columns
    assert np.isin(["X_0", "X_1", "X_2", "Y"], sample_vars).all() and len(sample_vars) == 4


def test_scm_sample():
    cn = build_scm_linandpoly(seed=0)
    n = 1000
    random.seed(0)
    scm_sample = cn.sample(n)
    sample = manual_standard_sample(
        n, scm_sample.values.dtype, list(cn.dag.nodes), seed=0
    )
    sample_order_scm = cn.get_variables()
    sample = sample[sample_order_scm]

    expectation_scm = scm_sample.mean(0)
    expectation_manual = sample.mean(0)
    exp_diff = (expectation_manual - expectation_scm).abs()
    assert (exp_diff.values < 0.1).all()


def test_scm_intervention():
    seed = 0
    cn = build_scm_linandpoly(seed=seed)

    # do the intervention
    cn.intervention(
        {
            "X_3": {
                "parents": ["X_0", "Y"],
                "assignment": "N + 3.3 * X_0 + 3.3 * Y",
                "noise": Normal("N", 5, 2),
            }
        }
    )
    n = 100

    scm_sample_interv = cn.sample(n)
    sample = np.empty((n, 5), dtype=scm_sample_interv.values.dtype)
    rng = random.Random(seed)
    rng.normalvariate(0, 1)
    noise_func = lambda size: np.array(
        [rng.normalvariate(mu=0, sigma=1) for _ in range(size)]
    )
    sample[:, 0] = noise_func(n)
    sample[:, 1] = 1 + noise_func(n) + 2 * sample[:, 0] ** 2
    sample[:, 2] = 1 + noise_func(n) + 3 * sample[:, 0] + 2 * sample[:, 1]
    sample[:, 4] = (
            noise_func(n)
            + Polynomial([0, 0, 0.05])(sample[:, 0])
            + Polynomial([0, 2])(sample[:, 2])
    )
    sample[:, 3] = np.asarray(
        list(sample_iter(Normal("N", 5, 2), numsamples=n))
    ) + 3.3 * (sample[:, 0] + sample[:, 4])
    sample = pd.DataFrame(sample, columns=list(cn.dag.nodes))
    manual_sample = sample[cn.get_variables()]

    manual_mean = manual_sample.mean(0)
    scm_mean = scm_sample_interv.mean(0)
    exp_diff = (manual_mean - scm_mean).abs().values
    assert (exp_diff < 0.1).all()

    # from here on the cn should work as normal again
    cn.undo_intervention()

    # reseeding needs to happen as the state of the initial noise distributions is further advanced than
    # a newly seeded noise by noise()
    cn.reseed(0)

    manual_sample = manual_standard_sample(
        n, scm_sample_interv.values.dtype, list(cn.dag.nodes), 0
    )[cn.get_variables()]
    new_cn_sample = cn.sample(n)
    manual_mean = manual_sample.mean(0)
    scm_mean = new_cn_sample.mean(0)
    exp_diff = (manual_mean - scm_mean).abs().values
    assert (exp_diff < 1e-1).all()


def test_scm_dointervention():
    seed = 0
    cn = build_scm_linandpoly(seed=seed)
    n = 100
    standard_sample = cn.sample(n, seed=seed)
    # do the intervention
    cn.do_intervention(["X_2"], [4])
    sample = cn.sample(n)
    assert (sample["X_2"] == 4).all()
    # from here on the cn should work as normal again
    cn.undo_intervention()
    new_sample = cn.sample(n, seed=seed)
    diff = (standard_sample.mean(0) - new_sample.mean(0)).abs().values
    assert (diff == 0.0).all()


def test_reproducibility():
    cn = build_scm_linandpoly()
    # cn.reseed(1)
    n = 20
    sample = cn.sample(n, seed=1)
    sample2 = cn.sample(n, seed=1)
    assert (sample.to_numpy() == sample2.to_numpy()).all()
