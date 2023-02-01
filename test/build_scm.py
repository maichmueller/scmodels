from scmodels import SCM
from sympy.stats import *


def build_scm_from_assignmentmap(seed=0):
    cn = SCM(
        assignments={
            "X_0": ("N", Normal("N", 0, 1)),
            "X_1": ("1 + X_0 + N", Normal("N", 0, 1)),
            "X_3": ("0.3 * X_1 + X_0 + N", Normal("N", 0, 1)),
        },
        variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$", "X_3": "$X_3$"},
        seed=seed,
    )
    return cn


def build_scm_from_assignmentstrs(seed=0):
    cn = SCM(
        assignments=[
            "X_0 = N, N ~ Normal(mean=0, std=1)",
            "X_1 = 1 + X_0 + N, N ~ Normal(mean=0, std=1)",
            "X_3 = 0.3 * X_1 + X_0 + N, N ~ Normal(mean=0, std=1)",
        ],
        variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$", "X_3": "$X_3$"},
        seed=seed,
    )
    return cn


def build_scm_from_functionalmap(seed=0):
    cn = SCM(
        assignments={
            "X_0": ([], lambda n: n, Normal("P", mean=0, std=1)),
            "X_1": (["X_0"], lambda n, x0: 1 + x0 + n, Normal("D", mean=0, std=1)),
            "X_3": (
                ["X_0", "X_1"],
                lambda n, x0, x1: 0.3 * x1 + x0 + n,
                Normal("M", mean=0, std=1),
            ),
        },
        variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$", "X_3": "$X_3$"},
        seed=seed,
    )
    return cn


def build_scm_simple(seed=0):
    cn = SCM(
        assignments={
            "X_0": ("N", Normal("N", 0, 1)),
            "X_1": ("1 + X_0 + N", Normal("N", 0, 1)),
            "X_2": ("1 + 3*X_0 + N", Normal("N", 0, 1)),
            "X_3": ("0.3 * X_1 + N", Normal("N", 0, 1)),
            "Y": ("3 + 5 * X_3 + N", Normal("N", 0, 1)),
            "X_4": ("3 + 9 * Y + N", Normal("N", 0, 1)),
            "X_5": ("3 - 2.7 * X_3 + N", Normal("N", 0, 1)),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
            "X_4": "$X_4$",
            "X_5": "$X_5$",
        },
        seed=seed,
    )
    return cn


def build_scm_medium(seed=0):
    cn = SCM(
        assignments={
            "X_0": ("N", Normal("N", 0, 1)),
            "X_1": ("N + 1 + X_0", Normal("N", 0, 1)),
            "X_2": ("N + 1 + 0.8 * X_0 - 1.2 * X_1", Normal("N", 0, 1)),
            "X_3": ("N + 1 + 0.3 * X_1 + 0.4 * X_2", Normal("N", 0, 1)),
            "Y": ("0.67 + X_3 + N - X_0", Normal("N", 0, 1)),
            "X_4": ("N + 1.2 - 0.7 * Y", Normal("N", 0, 1)),
            "X_5": ("N + 0.5 - 0.7 * Y", Normal("N", 0, 1)),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
            "X_4": "$X_4$",
            "X_5": "$X_5$",
        },
        seed=seed,
    )
    return cn


def build_scm_linandpoly(seed=0):
    cn = SCM(
        assignments={
            "X_0": ("N", Normal("N", 0, 1)),
            "X_1": ("N + 1 + 2 * (X_0 ** 2)", Normal("N", 0, 1)),
            "X_2": ("N + 1 + 3 * X_0 + 2 * X_1", Normal("N", 0, 1)),
            "X_3": (
                "N + X_1 + 0.5 * sqrt(abs(X_1)) + 4 * log(abs(X_2))",
                Normal("N", 0, 1),
            ),
            "Y": ("N + 0.05 * (X_0 ** 2) + 2 * X_2", Normal("N", 0, 1)),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
        },
        seed=seed,
    )
    return cn


def build_first_readme_example(seed=0):
    assignment_seq = [
        "Z = M, M ~ LogLogistic(alpha=1, beta=1)",
        "X = N * 3 * Z ** 2, N ~ LogNormal(mean=1, std=1)",
        "Y = P + 2 * Z + sqrt(X), P ~ Normal(mean=2, std=1)",
        "V = N**P + X, N ~ Normal(0,1) / P ~ Bernoulli(0.5)",
        "W = exp(T) - log(M) * N + Y, M ~ Exponential(1) / T ~ StudentT(0.5) / N ~ Normal(0, 2)",
    ]

    myscm = SCM(assignment_seq, seed=seed)
    return myscm


def build_second_readme_example(seed=0):
    from sympy.stats import LogLogistic, LogNormal, Normal, Bernoulli

    assignment_map = {
        "Z": ("M", LogLogistic("M", alpha=1, beta=1)),
        "X": (
            "N * 3 * Z ** 2",
            LogNormal("N", mean=1, std=1),
        ),
        "Y": (
            "P + 2 * Z + sqrt(X)",
            Normal("P", mean=2, std=1),
        ),
        "V": ("N**P + X", [Normal("N", 0, 1), Bernoulli("P", 0.5)]),
    }

    myscm = SCM(assignment_map, seed=seed)
    return myscm


def build_third_readme_example(seed=0):
    import numpy as np

    def y_assignment(p, z, x):
        return p + 2 * z + np.sqrt(x)

    functional_map = {
        "Z": ([], lambda m: m, LogLogistic("M", alpha=1, beta=1)),
        "X": (
            ["Z"],
            lambda n, z: n * 3 * z**2,
            LogNormal("N", mean=1, std=1),
        ),
        "Y": (
            ["Z", "X"],
            y_assignment,
            Normal("P", mean=2, std=1),
        ),
    }

    myscm = SCM(functional_map, seed=seed)
    return myscm
