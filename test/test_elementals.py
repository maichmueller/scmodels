from scm.functionals import *


def test_identity():
    ident = Identity(var=["X"])
    arg = np.arange(20).reshape(20, 1)
    assert (ident(arg) == arg).all()
    assert ident.arg_names() == set(["X"])


def test_constant():
    const = Constant(np.e, var=["X"])
    arg = np.arange(20).reshape(20, 1)
    expected = np.array([np.e] * 20).reshape(20, 1)
    assert (const(arg) == expected).all()


def test_affine():
    affine = Affine(2, 3, -1, var=["X", "Y"])
    arg = np.arange(40).reshape(20, 2)
    expected = 2 + arg[:, 0] * 3 + arg[:, 1] * (-1)
    assert (affine(arg[:, 0], arg[:, 1]) == expected).all()


def test_power():
    power_func = Power(exponent=np.e, var=["X"])
    arg = np.arange(20).reshape(20, 1)
    expected = np.power(arg, np.e)
    assert (power_func(arg) == expected).all()


def test_root():
    root_func = Root(nth_root=9, var=["X"])
    arg = np.arange(20).reshape(20, 1)
    expected = np.power(arg, 1 / 9)
    assert (root_func(arg) == expected).all()


def test_exp():
    exp = Exp(base=9, var=["X"])
    arg = np.arange(20).reshape(20, 1)
    expected = np.exp(arg * np.log(9))
    assert (exp(arg) == expected).all()


def test_log():
    log_func = Log(base=9, var=["X"])
    arg = np.arange(20).reshape(20, 1) + 1
    expected = np.log(arg) / np.log(9)
    assert (log_func(arg) == expected).all()


def test_sin():
    sin_func = Sin(var=["X"])
    arg = np.arange(20).reshape(20, 1)
    expected = np.sin(arg)
    assert (sin_func(arg) == expected).all()


def test_cos():
    cos = Cos(var=["X"])
    arg = np.arange(20).reshape(20, 1)
    expected = np.cos(arg)
    assert (cos(arg) == expected).all()


def test_tan():
    tan = Tan(var=["X"])
    arg = np.arange(20).reshape(20, 1)
    expected = np.tan(arg)
    assert (tan(arg) == expected).all()


def test_arcsin():
    arcsin = Arcsin(var=["X"])
    arg = np.random.default_rng(0).random(20) * 2 - 1
    expected = np.arcsin(arg)
    assert (arcsin(arg) == expected).all()


def test_arccos():
    arccos = Arccos(var=["X"])
    arg = np.random.default_rng(0).random(20) * 2 - 1
    expected = np.arccos(arg)
    assert (arccos(arg) == expected).all()


def test_arctan():
    arctan = Arctan(var=["X"])
    arg = np.random.default_rng(0).random(20) * 2 - 1
    expected = np.arctan(arg)
    assert (arctan(arg) == expected).all()


def test_sinh():
    sinh = Sinh(var=["X"])
    arg = np.arange(20).reshape(20, 1)
    expected = np.sinh(arg)
    assert (sinh(arg) == expected).all()


def test_cosh():
    cosh = Cosh(var=["X"])
    arg = np.arange(20).reshape(20, 1)
    expected = np.cosh(arg)
    assert (cosh(arg) == expected).all()


def test_tanh():
    tanh = Tanh(var=["X"])
    arg = np.arange(20).reshape(20, 1)
    expected = np.tanh(arg)
    assert (tanh(arg) == expected).all()


def test_arcsinh():
    arcsinh = Arcsinh(var=["X"])
    arg = np.arange(20).reshape(20, 1) + 1
    expected = np.arcsinh(arg)
    assert (arcsinh(arg) == expected).all()


def test_arccosh():
    arccosh = Arccosh(var=["X"])
    arg = np.arange(20).reshape(20, 1) + 1
    expected = np.arccosh(arg)
    assert (arccosh(arg) == expected).all()


def test_arctanh():
    arctanh = Arctanh(var=["X"])
    arg = np.random.default_rng(0).random(20) * 2 - 1
    expected = np.arctanh(arg)
    assert (arctanh(arg) == expected).all()
