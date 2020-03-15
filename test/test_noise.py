from scm.functionals import *
from scm.noise import *


def test_single_noise():
    seed = 0
    f = Affine(
        0,
        1,
        2,
        noise_generator=NoiseGenerator("standard_normal", seed=seed),
        additive_noise=True,
    )
    x, y = np.split(np.arange(50).reshape(25, 2), 2, axis=1)

    evaluated = f(x, y)
    expected = x + y * 2 + np.random.default_rng(seed).normal(25)
    assert (evaluated == expected).all()
