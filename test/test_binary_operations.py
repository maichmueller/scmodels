from scm.functionals import *


def test_addition():
    f1 = Affine(0, 1, 2)
    f2 = Affine(0, 3, 5)
    f3 = f1 + f2
    arg = 