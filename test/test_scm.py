import random
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
from test.build_scm import *
from scmodels.parser import parse_assignments, extract_parents


def manual_sample_linandpoly(n, dtype, names, seed):
    def x1(n, x0):
        return 1 + n + Polynomial([0, 0, 2])(x0)

    def x2(n, x0, x1):
        return 1 + n + 3 * x0 + 2 * x1

    def x3(n, x1, x2):
        return n + x1 + 0.5 * np.sqrt(x1.abs()) + 4 * np.log(x2.abs())

    def y(n, x0, x2):
        return n + Polynomial([0, 0, 0.05])(x0) + 2 * x2

    rng = np.random.default_rng(seed)
    noise_func = lambda size: rng.normal(loc=0, scale=1, size=size)
    sample = pd.DataFrame(np.empty((n, 5), dtype=dtype), columns=names)
    sample.loc[:, "X_0"] = noise_func(n)
    sample.loc[:, "X_1"] = x1(noise_func(n), sample["X_0"])
    sample.loc[:, "X_2"] = x2(noise_func(n), sample["X_0"], sample["X_1"])
    sample.loc[:, "X_3"] = x3(noise_func(n), sample["X_1"], sample["X_2"])
    sample.loc[:, "Y"] = y(noise_func(n), sample["X_0"], sample["X_2"])
    return sample


def manual_sample_medium(n, dtype, names, seed):
    def x1(n, x0):
        return 1 + n + x0

    def x2(n, x0, x1):
        return 1 + n + 0.8 * x0 - 1.2 * x1

    def x3(n, x1, x2):
        return n + x1 + 0.3 * x1 + 0.4 * x2

    def y(n, x0, x3):
        return 0.67 + n + x3 - x0

    def x4(n, y):
        return 1.2 + n - 0.7 * y

    def x5(n, y):
        return n + 0.5 - 0.7 * y

    rng = np.random.default_rng(seed)
    noise_func = lambda size: rng.normal(loc=0, scale=1, size=size)
    sample = pd.DataFrame(np.empty((n, len(names)), dtype=dtype), columns=names)
    sample.loc[:, "X_0"] = noise_func(n)
    sample.loc[:, "X_1"] = x1(noise_func(n), sample["X_0"])
    sample.loc[:, "X_2"] = x2(noise_func(n), sample["X_0"], sample["X_1"])
    sample.loc[:, "X_3"] = x3(noise_func(n), sample["X_1"], sample["X_2"])
    sample.loc[:, "Y"] = y(noise_func(n), sample["X_0"], sample["X_3"])
    sample.loc[:, "X_4"] = x4(noise_func(n), sample["Y"])
    sample.loc[:, "X_5"] = x5(noise_func(n), sample["Y"])
    return sample


def same_elements(list1, list2):
    return np.isin(list1, list2).all() and np.isin(list2, list1).all()


def test_parsing():
    test_str = "Z_1 = Noise + 2*log(Y), Noise ~ Normal(0,1)"
    func_map = parse_assignments([test_str])
    assert func_map["Z_1"][0] == "Noise + 2*log(Y)"
    assert str(func_map["Z_1"][1]) == "Noise"
    assert extract_parents(func_map["Z_1"][0], "Noise") == ["Y"]

    test_str = "X = N + sqrt(X_45 ** M + 342 * 2) / (  43 * FG_2) + P, N ~ Normal(0,1)"
    func_map = parse_assignments([test_str])
    assert func_map["X"][0] == "N + sqrt(X_45 ** M + 342 * 2) / (  43 * FG_2) + P"
    assert str(func_map["X"][1]) == "N"
    assert same_elements(
        extract_parents(func_map["X"][0], "N"), ["X_45", "M", "FG_2", "P"]
    )


def test_scm_from_strings():
    scm = SCM(
        [
            "X = N, N ~ Normal(0,1)",
            "Y_0 = M + 2 * exp(X), M ~ StudentT(2)",
            "Y_1 = M + 2 * exp(sqrt(X)), M ~ Normal(0, 0.1)",
            "Z = P * sqrt(Y_0), P ~ Exponential(5.3)",
        ]
    )
    assert same_elements(scm.get_variables(), ["X", "Y_0", "Y_1", "Z"])


def test_scm_build_from_assignmentmap():
    cn = build_scm_from_assignmentmap()
    nodes_in_graph = sorted(cn.get_variables())
    assert same_elements(nodes_in_graph, ["X_0", "X_1", "X_3"])
    assert same_elements(cn["X_3"].parents, ["X_0", "X_1"])
    assert same_elements(cn["X_1"].parents, ["X_0"])


def test_scm_build_from_functionalmap():
    cn = build_scm_from_functionalmap()
    nodes_in_graph = sorted(cn.get_variables())
    assert same_elements(nodes_in_graph, ["X_0", "X_1", "X_3"])
    assert same_elements(cn["X_3"].parents, ["X_0", "X_1"])
    assert same_elements(cn["X_1"].parents, ["X_0"])


def test_scm_build_from_assignmentstrs():
    cn = build_scm_from_assignmentstrs()
    nodes_in_graph = sorted(cn.get_variables())
    assert same_elements(nodes_in_graph, ["X_0", "X_1", "X_3"])
    assert same_elements(cn["X_3"].parents, ["X_0", "X_1"])
    assert same_elements(cn["X_1"].parents, ["X_0"])


def test_scm_builds_equal_sampling():
    cn1 = build_scm_from_assignmentmap()
    cn2 = build_scm_from_functionalmap()
    cn3 = build_scm_from_assignmentstrs()
    cn1.seed(0)
    cn2.seed(0)
    cn3.seed(0)
    n = 10
    sample1 = cn1.sample(n)
    sample2 = cn2.sample(n)
    sample3 = cn3.sample(n)
    samples = sample1, sample2, sample3
    for i in range(2):
        for var in cn1.get_variables():
            assert same_elements(samples[i][var].round(4), samples[i + 1][var].round(4))


def test_scm_sample_partial():
    cn = build_scm_linandpoly(seed=0)
    n = 1000
    random.seed(0)
    sample_vars = cn.sample(n, ["X_0"]).columns
    assert same_elements(sample_vars, ["X_0"])

    sample_vars = cn.sample(n, ["X_2"]).columns
    assert same_elements(["X_0", "X_1", "X_2"], sample_vars)

    sample_vars = cn.sample(n, ["Y"]).columns
    assert same_elements(["X_0", "X_1", "X_2", "Y"], sample_vars)


def test_scm_sample():
    n = 10000
    cn = build_scm_linandpoly(seed=0)
    scm_sample = cn.sample(n)
    sample = manual_sample_linandpoly(
        n, scm_sample.values.dtype, list(cn.dag.nodes), seed=0
    )
    sample_order_scm = cn.get_variables()
    sample = sample[sample_order_scm]

    expectation_scm = scm_sample.mean(0)
    expectation_manual = sample.mean(0)
    exp_diff = (expectation_manual - expectation_scm).abs()
    assert (exp_diff.values < 0.1).all()

    cn = build_scm_medium(seed=0)
    scm_sample = cn.sample(n)
    sample = manual_sample_medium(
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
        {"X_3": (None, "N + 2.3 * X_0 + 2.3 * Y", Normal("N", 5, 2))}
    )
    n = 10000

    scm_sample_interv = cn.sample(n)
    rng = np.random.default_rng(seed)
    sample = manual_sample_linandpoly(
        n, dtype=np.float, names=cn.get_variables(), seed=cn.rng_state
    )
    sample["X_3"] = rng.normal(loc=5, scale=2, size=n) + 2.3 * (
            sample["X_0"] + sample["Y"]
    )
    sample = sample[cn.get_variables()]

    manual_mean = sample.mean(0)
    scm_mean = scm_sample_interv.mean(0)
    exp_diff = (manual_mean - scm_mean).abs().values
    assert (exp_diff < 0.5).all()

    # from here on the cn should work as normal again
    cn.undo_intervention()

    # reseeding needs to happen as the state of the initial noise distributions is further advanced than
    # a newly seeded noise by noise()
    cn.seed(0)

    manual_sample = manual_sample_linandpoly(
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
    cn.do_intervention([("X_2", 4)])
    sample = cn.sample(n)
    assert (sample["X_2"] == 4).all()
    # from here on the cn should work as normal again
    cn.undo_intervention()
    new_sample = cn.sample(n, seed=seed)
    diff = (standard_sample.mean(0) - new_sample.mean(0)).abs().values
    assert (diff == 0.0).all()


def test_scm_softintervention():
    seed = 0
    cn = build_scm_linandpoly(seed=seed)
    n = 100
    standard_sample = cn.sample(n, seed=seed)
    # do the intervention
    cn.soft_intervention([("X_2", FiniteRV(str(cn["X_2"].noise), density={0: 1.}))])
    sample = cn.sample(n)
    assignment = cn["X_2"].assignment
    manual_x2 = assignment(0, sample["X_0"], sample["X_1"])
    diff = (sample["X_2"] - manual_x2).abs()
    assert np.all(diff == 0.0)

    # from here on the cn should work as normal again
    cn.undo_intervention()
    new_sample = cn.sample(n, seed=seed)
    diff = (standard_sample.mean(0) - new_sample.mean(0)).abs().values
    assert np.all(diff == 0.0)


def test_sample_iter():
    cn = build_scm_from_assignmentmap()
    samples = {var: [] for var in cn.get_variables()}
    rng1 = np.random.default_rng(seed=0)
    rng2 = np.random.default_rng(seed=0)
    iterator = cn.sample_iter(samples, seed=rng1)
    n = 1000
    for _ in range(n):
        next(iterator)
    standard_sample = cn.sample(n, seed=rng2)
    samples = pd.DataFrame.from_dict(samples)
    diff = (standard_sample.mean(0) - samples.mean(0)).abs().values
    assert (diff < 1e-1).all()


def test_reproducibility():
    cn = build_scm_linandpoly()
    n = 20
    sample = cn.sample(n, seed=1)
    sample2 = cn.sample(n, seed=1)
    assert (sample.to_numpy() == sample2.to_numpy()).all()


def test_none_noise():
    cn = SCM(["X = 1", "Y = N + X, N ~ Normal(0,1)"], seed=0)
    n = 10
    sample = cn.sample(n)
    manual_y = np.random.default_rng(0).normal(size=n) + 1
    assert (sample["X"] == 1).all()
    assert (sample["Y"] == manual_y).all()

    sample = {var: [] for var in cn.get_variables()}
    sampler = cn.sample_iter(sample, seed=0)
    list(next(sampler) for _ in range(n))
    sample = pd.DataFrame.from_dict(sample)
    manual_y = np.random.default_rng(0).normal(size=n) + 1
    assert (sample["X"] == 1).all()
    assert (sample["Y"] == manual_y).all()
