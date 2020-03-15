from scm.functionals import *
import numpy as np


def test_addition():
    f1 = Affine(1, 1, 2)
    f2 = Affine(-3, 3, 5)
    f3 = f1 + f2
    arg = np.random.default_rng(0).random((10, 2))
    evaluated = f3(arg[:, 0], arg[:, 1])
    assert ((evaluated - (arg[:, 0] * 4 + arg[:, 1] * 7 + -2)).round(10) == 0).all()


def test_subtraction():
    f1 = Affine(1, 1, 2)
    f2 = Affine(-3, 3, 5)
    f3 = f1 - f2
    arg = np.random.default_rng(0).random((10, 2))
    evaluated = f3(arg[:, 0], arg[:, 1])
    assert (
        (evaluated - (arg[:, 0] * (-2) + arg[:, 1] * (-3) + 4)).round(10) == 0
    ).all()


def test_multiplication():
    f1 = Affine(1, 1, 2)
    f2 = Affine(-3, 3, 5)
    f3 = f1 * f2
    arg = np.random.default_rng(0).random((10, 2))
    evaluated = f3(arg[:, 0], arg[:, 1])
    expected = (
        1 * (-3 + arg[:, 0] * 3 + arg[:, 1] * 5)
        + 1 * arg[:, 0] * (-3 + arg[:, 0] * 3 + arg[:, 1] * 5)
        + 2 * arg[:, 1] * (-3 + arg[:, 0] * 3 + arg[:, 1] * 5)
    )
    assert ((evaluated - expected).round(10) == 0).all()


def test_multiples():
    f1 = Affine(1, 1, 2, var=["X", "Z"])
    f2 = Exp(3, var=["X"])
    f3 = Sin(var=["Z"])
    f4 = Power(4, var=["Y"])
    f5 = (f2 @ f1) * f3 + f4
    X, Z, Y = list(
        map(lambda x: x.flatten(), np.split(np.arange(75).reshape(25, 3), 3, axis=1))
    )
    f2_of_f1 = np.exp(np.log(3) * (1 + X * 1 + Z * 2))
    expected = f2_of_f1 * np.sin(Z) + Y ** 4

    # positional dispatch (dangerous as variable names may be shifted around)
    evaluated = f5(X, Z, Y)
    assert ((evaluated - expected).round(10) == 0).all()

    # test with kwarg dispatch
    evaluated = f5(X=X, Y=Y, Z=Z)
    assert ((evaluated - expected).round(10) == 0).all()
