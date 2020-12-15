from scm import SCM
from sympy.stats import *

def build_scm_minimal(seed=0):
    cn = SCM(
        assignments={
            "X_0": ("N", Normal("N", 0, 1)),
            "X_1": ("1 + X_0 + N", Normal("N", 0, 1)),
            "X_3": ("0.3 * X_1 + N", Normal("N", 0, 1))
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_3": "$X_3$",
        },
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
