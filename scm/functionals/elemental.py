from typing import Optional, Collection, Hashable, Tuple

from .functionals import Functional
import numpy as np


class Identity(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return arg

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return list(self.get_arg_names())[0]


class Constant(Functional):
    def __init__(self, const_value: float, **base_kwargs):
        self.const_value = const_value
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.ones_like(arg) * self.const_value

    def __len__(self):
        return len(self.get_arg_names())

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return str(self.const_value)


class Affine(Functional):
    r"""
    The affine Functional function of the form:
        f(X_S, N) = offset + \sum_{i \in S} a_i * X_i + noise_coeff * N
    """

    def __init__(
        self,
        offset: float = 0.0,
        *coefficients,
        var: Optional[Collection[Hashable]] = None,
        **base_kwargs,
    ):
        self.offset = offset
        self.coefficients = np.asarray(coefficients)

        if var is None:
            var = tuple(str(i) for i in range(len(coefficients)))
        base_kwargs.update(dict(var=var))
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, *args, **kwargs):
        return self.offset + self.coefficients @ args

    def __len__(self):
        return 1 + len(self.args_to_position)

    def function_str(self, variable_names=None):
        rep = ""
        if self.offset != 0:
            rep += str(round(self.offset, 2))
        for i, c in enumerate(self.coefficients):
            if c != 0:
                rep += f" + {round(c, 2)} {variable_names[i + 1]}"
        return rep


class Power(Functional):
    def __init__(self, exponent: float, **base_kwargs):
        self.exponent: float = exponent
        base_kwargs.update(dict(is_elemental=True))
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: Tuple[np.ndarray], **kwargs):
        return np.power(arg, self.exponent)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"{list(self.get_arg_names())[0]} ** {self.exponent}"


class Root(Functional):
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

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"{self.ordinal}-root({list(self.get_arg_names())[0]})"


class Exp(Functional):
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

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"exp({list(self.get_arg_names())[0]})"


class Log(Functional):
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

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"exp({list(self.get_arg_names())[0]})"


class Sin(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.sin(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"sin({list(self.get_arg_names())[0]})"


class Cos(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.cos(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"cos({list(self.get_arg_names())[0]})"


class Tan(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.tan(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"tan({list(self.get_arg_names())[0]})"


class Arcsin(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arcsin(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"arcsin({list(self.get_arg_names())[0]})"


class Arccos(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arccos(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"arccos({list(self.get_arg_names())[0]})"


class Arctan(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arctan(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"arctan({list(self.get_arg_names())[0]})"


class Sinh(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.sinh(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"sinh({list(self.get_arg_names())[0]})"


class Cosh(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.cosh(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"cosh({list(self.get_arg_names())[0]})"


class Tanh(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.tanh(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"tanh({list(self.get_arg_names())[0]})"


class Arcsinh(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arcsinh(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"arcsinh({list(self.get_arg_names())[0]})"


class Arccosh(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arccosh(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"arccosh({list(self.get_arg_names())[0]})"


class Arctanh(Functional):
    def __init__(self, **base_kwargs):
        base_kwargs["is_elemental"] = True
        super().__init__(**base_kwargs)

    @Functional.call_arg_handler
    def __call__(self, arg: np.ndarray, **kwargs):
        return np.arctanh(arg)

    def __len__(self):
        return 1

    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
        return f"arctanh({list(self.get_arg_names())[0]})"
