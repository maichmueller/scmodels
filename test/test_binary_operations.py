from scm.functionals import *
import numpy as np


def test_addition():
    f1 = Affine(1, 1, 2)
    f2 = Affine(-3, 3, 5)
    f3 = f1 + f2
    arg = np.random.default_rng(0).random((10, 2))
    evaluated = f3(arg[:, 0], arg[:, 1])
    assert ((evaluated - (arg[:, 0] * 4 + arg[:, 1] * 7 + -2)).round(10) == 0).all()


def test_addition():
    f1 = Affine(1, 1, 2)
    f2 = Affine(-3, 3, 5)
    f3 = f1 + f2
    arg = np.random.default_rng(0).random((10, 2))
    evaluated = f3(arg[:, 0], arg[:, 1])
    assert ((evaluated - (arg[:, 0] * 4 + arg[:, 1] * 7 + -2)).round(10) == 0).all()


