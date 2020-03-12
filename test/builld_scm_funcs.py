from scm import AffineFunctional, PolynomialFunctional, NoiseGenerator, SCM


def build_scm_simple(seed=0):
    cn = SCM(
        functional_map={
            "X_0": (
                [],
                AffineFunctional(1),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_1": (
                ["X_0"],
                AffineFunctional(1, 1, 2),
                NoiseGenerator("standard_normal", seed=seed+1),
            ),
            "X_2": (
                ["X_0"],
                AffineFunctional(1, 1, 3),
                NoiseGenerator("standard_normal", seed=seed+2),
            ),
            "X_3": (
                ["X_1"],
                AffineFunctional(1, 0, 0.3),
                NoiseGenerator("standard_normal", seed=seed+3),
            ),
            "Y": (
                ["X_3"],
                AffineFunctional(1, 3, 5),
                NoiseGenerator("standard_normal", seed=seed+4),
            ),
            "X_4": (
                ["Y"],
                AffineFunctional(1, 3, 9),
                NoiseGenerator("standard_normal", seed=seed+5),
            ),
            "X_5": (
                ["X_3"],
                AffineFunctional(1, 3, -2.7),
                NoiseGenerator("standard_normal", seed=seed+6),
            ),
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


def build_scm_medium(seed=0):
    cn = SCM(
        functional_map={
            "X_0": (
                [],
                AffineFunctional(1),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_1": (
                ["X_0"],
                AffineFunctional(1, 1, 1),
                NoiseGenerator("standard_normal", seed=seed+1),
            ),
            "X_2": (
                ["X_0", "X_1"],
                AffineFunctional(1, 1, 0.8, -1.2),
                NoiseGenerator("standard_normal", seed=seed+2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                AffineFunctional(1, 0, 0.3, 0.4),
                NoiseGenerator("standard_normal", seed=seed+3),
            ),
            "Y": (
                ["X_3", "X_0"],
                AffineFunctional(1, 0.67, 1, -1),
                NoiseGenerator("standard_normal", seed=seed+4),
            ),
            "X_4": (
                ["Y"],
                AffineFunctional(1, 1.2, -.7),
                NoiseGenerator("standard_normal", seed=seed+5),
            ),
            "X_5": (
                ["X_3", "Y"],
                AffineFunctional(1, 0.5, -.7, 0.4),
                NoiseGenerator("standard_normal", seed=seed+6),
            ),
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
        functional_map={
            "X_0": (
                [],
                AffineFunctional(1),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_1": (
                ["X_0"],
                AffineFunctional(1, 1, 2),
                NoiseGenerator("standard_normal", seed=seed+1),
            ),
            "X_2": (
                ["X_0", "X_1"],
                AffineFunctional(1, 1, 3, 2),
                NoiseGenerator("standard_normal", seed=seed+2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                PolynomialFunctional([0, 1], [0, 1, 0.5], [0, 0, 4]),
                NoiseGenerator("standard_normal", seed=seed+3),
            ),
            "Y": (
                ["X_0", "X_2"],
                PolynomialFunctional([0, 1], [0, 0, 1.5], [0, 2]),
                NoiseGenerator("standard_normal", seed=seed+4),
            ),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
        },
    )
    return cn
