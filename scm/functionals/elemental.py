from typing import Optional, Collection, Tuple

from .functionals import Functional
import numpy as np
from abc import ABC
from typing import Callable
import functools


class ElementalFunctional(Functional, ABC):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @staticmethod
    def default_names_handler(func: Callable):
        @functools.wraps(func)
        def wrapped_fstr(
            self, variable_names: Optional[Collection[str]] = None, *args, **kwargs
        ):
            if variable_names is None:
                variable_names = self.arg_names()
            return func(self, variable_names, *args, **kwargs)

        return wrapped_fstr


class Identity(ElementalFunctional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return arg

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return variable_names[0]


class Constant(ElementalFunctional):
    def __init__(self, const_value: float, **base_kwargs):
        self.const_value = const_value
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.ones_like(arg) * self.const_value

    def __len__(self):
        return len(self.arg_names())

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return str(self.const_value)


class Affine(ElementalFunctional):
    r"""
    The affine Functional function of the form:
        f(X_S, N) = offset + \sum_{i \in S} a_i * X_i + noise_coeff * N
    """

    def __init__(
        self, offset: float = 0.0, *coefficients, **base_kwargs,
    ):
        self.offset = offset
        self.coefficients = np.asarray(coefficients)
        # add the number of variables to the var names list if they haven't already been supplied
        base_kwargs["var"] = base_kwargs.pop("var", len(self.coefficients))
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, *args, **kwargs):
        return self.offset + self.coefficients @ args

    def __len__(self):
        return len(self.args_to_position)

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        rep = ""
        if self.offset != 0:
            rep += str(round(self.offset, 2))
        uses_noise = self.accepts_noise
        for i, c in enumerate(self.coefficients, 1 if uses_noise else 0):
            if c != 0:
                rep += f" + {round(c, 2)} {variable_names[i]}"
        return rep


class Power(ElementalFunctional):
    def __init__(self, exponent: float, **base_kwargs):
        self.exponent: float = exponent
        base_kwargs.update(dict(is_elemental=True))
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: Tuple[np.ndarray], **kwargs):
        return np.power(arg, self.exponent)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        if variable_names is None:
            variable_names = self.arg_names()
        variable_names = variable_names[0]
        return f"{variable_names[0]} ** {self.exponent}"


class Root(ElementalFunctional):
    def __init__(self, nth_root: int, **base_kwargs):
        self.nth_root: int = nth_root
        self.ordinal = "%d%s" % (
            self.nth_root,
            {1: "st", 2: "nd", 3: "rd"}.get(
                self.nth_root if self.nth_root < 20 else self.nth_root % 10, "th"
            ),
        )
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.power(arg, 1 / self.nth_root)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"{self.ordinal}-root({variable_names[0]})"


class Exp(ElementalFunctional):
    def __init__(self, base: float = np.e, **base_kwargs):
        self.base: float = np.log(base)
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        if self.base != 1:
            return np.exp(arg * self.base)
        else:
            return np.exp(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        if variable_names is None:
            variable_names = self.arg_names()
        variable_names = variable_names[0]
        return f"exp({variable_names})"


class Log(ElementalFunctional):
    def __init__(self, base: float = np.e, **base_kwargs):
        self.base_log: float = np.log(base)
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        if self.base_log != 1:
            return np.log(arg) / self.base_log
        else:
            return np.log(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"exp({variable_names[0]})"


class Sin(ElementalFunctional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.sin(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"sin({variable_names[0]})"


class Cos(ElementalFunctional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.cos(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"cos({variable_names[0]})"


class Tan(ElementalFunctional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.tan(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"tan({variable_names[0]})"


class Arcsin(ElementalFunctional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arcsin(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"arcsin({variable_names[0]})"


class Arccos(ElementalFunctional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arccos(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"arccos({variable_names[0]})"


class Arctan(ElementalFunctional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arctan(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"arctan({variable_names[0]})"


class Sinh(ElementalFunctional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.sinh(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"sinh({variable_names[0]})"


class Cosh(ElementalFunctional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.cosh(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"cosh({variable_names[0]})"


class Tanh(ElementalFunctional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.tanh(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"tanh({variable_names[0]})"


class Arcsinh(ElementalFunctional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arcsinh(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"arcsinh({variable_names[0]})"


class Arccosh(ElementalFunctional):
    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arccosh(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"arccosh({variable_names[0]})"


class Arctanh(ElementalFunctional):
    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arctanh(arg)

    def __len__(self):
        return 1

    @ElementalFunctional.default_names_handler
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"arctanh({variable_names[0]})"
