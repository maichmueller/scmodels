from scm.noise import Noise

import functools
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, TypeVar, Optional, List, Collection, Dict, Callable, Tuple


class Functional(ABC):
    """
    An abstract base class for functionals. Functionals are supposed to be functions with a fixed call scheme:
    The first positional argument will be the evaluation of a noise variable N.
     Any further inputs are the evaluations of parent variables, which the associated variable of the functional
     depends on.

     Any inheriting class will need to implement two functions (aside the __init__ constructor):
    - '__call__': A call method to evaluate the function.
    - 'str': A conversion of the functional to print out its mathematical behaviour in the form of:
        f(N, x_0, ..., x_k) = ...

    See details of these functions for further explanation.
    """

    def __init__(
        self,
        var: Union[Collection[str], int] = 1,
        noise_generator: Optional[Noise] = None,
        use_external_noise: Optional[bool] = False,
        additive_noise: bool = True,
        is_argument_noise: bool = False,
        noise_transformation: Callable = lambda x: x,
        is_elemental: bool = True,
    ):
        self.noise_gen = noise_generator
        self.use_external_noise = use_external_noise
        self.accepts_noise = noise_generator is not None or use_external_noise
        # whether we add the noise to the func or multiply it.
        self.additive_noise = additive_noise
        # whether the noise acts on the functionals' arguments or the functional itself ( f(N*x) vs N*f(x) )
        self.inner_noise = is_argument_noise
        # how the noise is treated inside the functional determined by a callable, that will be used when
        # noise is passed later on: N = self.noise_transformation(noise_array)
        if self.accepts_noise:
            # set the default noise ignore methods, 0 for additive and 1 for multiplicative noise
            if additive_noise:
                self.noise_transformation = lambda x: 0
            else:
                self.noise_transformation = lambda x: 1
        else:
            self.noise_transformation = noise_transformation
        # the names of the functionals' arguments
        self.has_named_args = False
        # the positions of the argument names.
        self.args_to_position: Dict[str, int] = dict()
        if isinstance(var, int):
            var = tuple(f"X{i}" for i in range(var))
        self.set_names_for_args(list(var))
        # whether the functional is a concatenation, addition, multiplication etc. of other functionals
        self.is_elemental = is_elemental

        if len(self) != 1 and self.inner_noise:
            raise ValueError(
                "Noise can only be applied to arguments if the argument is 1 dimensional."
            )

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        The call method to evaluate the functional (function). The interface only enforces an input of the noise
        variable for each inheriting class, as these are essential parts of an SCM, that can't be omitted.

        It is implicitly necessary in the current framework, that the implemented call method is vectorized, i.e. is
        able to evaluate numpy arrays of consistent shape correctly. This improves sampling from these functions,
        avoiding python loops in favor of numpy C calculations.

        Notes
        -----
        Every __call__ method in a child class should decorate their methods with the _call_decorator from the base
        `Functional` class, which then allows the usage of kwargs for named function argument calls and enables the
        usage of noise.

        Parameters
        ----------
        noise: float or np.ndarray,
            the input of the noise variable of a single sample or of an array of samples.
        args: any positional arguments needed by the subclass.
        kwargs: any keyword arguments needed by the subclass.

        Returns
        -------
        float or np.ndarray, the evaluated function for all provided inputs.
        """
        raise NotImplementedError(
            "Functional subclasses must provide '__call__' method."
        )

    def __len__(self):
        """
        Method to return the number of variables the functional takes.
        """
        return len(self.arg_names())

    def __add__(self, other):
        return AddFunctional(self, other)

    def __sub__(self, other):
        return SubFunctional(self, other)

    def __mul__(self, other):
        return MulFunctional(self, other)

    def __truediv__(self, other):
        return DivFunctional(self, other)

    def __matmul__(self, other):
        """
        This operator (@) is reserved for concatenation.
        """
        return CompositionFunctional(self, other)

    @abstractmethod
    def function_str(self, variable_names: Optional[Collection[str]] = None):
        """
        Method to convert the functional functor to console printable output of the form:
            f(N, x_0,...) = math_rep_of_func_as_string_here...

        Notes
        -----
        The inheriting class only needs to implement the actual functional part of the string!
        This means, that the prefix 'f(N, X_0, X1,...) =' is added by the base class (with correct number of variables),
        and the child class only needs to provide the string representation of the right hand side of the functional.

        Parameters
        ----------
        variable_names: (optional) List[str],
            a list of string names for each of the input variables in sequence.
            Each position will be i of the list will be the name of the ith positional variable argument in __call__.

        Returns
        -------
        str, the converted identifier of the function.
        """
        raise NotImplementedError(
            "Functional subclasses must provide 'function_str' method."
        )

    def __str__(self):
        return self.str()

    def str(self, variable_names: Optional[Collection[str]] = None):
        if variable_names is None:
            variable_names = self.arg_names()
        if self.accepts_noise:
            variable_names = ["N"] + list(variable_names)
        functional = self.function_str(variable_names)
        prefix = f"f({', '.join(variable_names)}) = "
        return prefix + functional

    def set_names_for_args(self, names_collection: Collection[str]):
        """
        Set the positional relation of input arguments with names.

        This method will help functional function calls later on to provide kwargs with the names of
        causal parents in the graph, without having specifically named these parents in the function
        definition of the functional already.
        In short: enables dynamic kwarg dispatch on functionals if desired.
        """
        for position, name in enumerate(names_collection):
            self.args_to_position[name] = position
        self.has_named_args = True

    def arg_names(self):
        return list(self.args_to_position.keys())

    @staticmethod
    def call_arg_handler(func):
        @functools.wraps(func)
        def wrapped__call__(
            self, *args, noise: Optional[Union[np.ndarray, float]] = None, **kwargs
        ):
            noise, args = self.prep_call_input(*args, noise=noise, **kwargs)
            output = func(self, *args)
            if noise is not None:
                output = self.sprinkle_noise(output, noise)
            return output

        return wrapped__call__

    def prep_call_input(
        self, *args, noise: Optional[Union[np.ndarray, float]] = None, **kwargs
    ):
        """
        This method will parse the provided args and kwargs to return only args, that have been
        rearranged according to the order previously set for named args.

        Combined passing of args and kwargs is forbidden and raises a `ValueError`.
        """
        if kwargs:
            if args:
                raise ValueError(
                    "Either only args or only kwargs can be provided to functional call."
                )

            # sanity check
            if not self.has_named_args:
                raise ValueError(
                    "Kwargs provided, but functional doesn't have named arguments."
                )

            filtered_kwargs_gen = (
                (key, arg)
                for (key, arg) in kwargs.items()
                if key in self.args_to_position
            )
            args = tuple(
                val
                for key, val in sorted(
                    filtered_kwargs_gen,
                    key=lambda key_val_pair: self.args_to_position[key_val_pair[0]],
                )
            )
        elif args:
            args = tuple(
                arg for arg, _ in zip(args, range(len(self)))
            )  # ignore too many args actively.

        if not self.accepts_noise and noise is not None:
            raise ValueError(
                "External noise was passed to functional, "
                "but functional doesn't accept noise."
            )

        if self.accepts_noise:
            if not self.use_external_noise:
                noise = [self.noise_gen(np.asarray(arg).shape[0]) for arg in args]
            noise = self.noise_transformation(noise)
            if self.inner_noise:
                # inner noise only makes sense for single argument functions, i.e. len(args) == 1
                if self.additive_noise:
                    args = tuple(n + arg for n, arg in zip(noise, args))
                else:
                    args = tuple(n * arg for n, arg in zip(noise, args))
                noise = None

        return noise, args

    def sprinkle_noise(
        self, output: Union[np.ndarray, float], noise: Union[np.ndarray, float]
    ):
        """
        Add the noise to the output, if the noise was meant as an output noise (in contrast to argument noise).
        """
        if not self.inner_noise:
            if self.additive_noise:
                output += noise
            else:
                output *= noise
        return output


class BinaryFunctional(Functional, ABC):
    def __init__(self, func1: Functional, func2: Functional):
        self.func1 = func1
        self.func2 = func2

        noise_gen = None
        if func1.noise_gen is not None and func2.noise_gen is not None:
            assert (
                func1.noise_gen.noise_description == func2.noise_gen.noise_description,
                "The sub functionals shouldn't differ in their expected noise.",
            )
            noise_gen = (
                func1.noise_gen if func1.noise_gen is not None else func2.noise_gen
            )

        for func in (func1, func2):
            if func.accepts_noise:
                func.use_external_noise = True

        f1_args = func1.arg_names()
        v_names = f1_args + [arg for arg in func2.arg_names() if arg not in f1_args]
        super().__init__(
            noise_generator=noise_gen,
            use_external_noise=False,
            var=v_names,
            is_elemental=False,
        )


class CompositionFunctional(BinaryFunctional):
    r"""
    The Linear Functional function of the form:
        f(X_S, N) = offset + \sum_{i \in S} a_i * X_i + noise_coeff * N
    """

    def __init__(self, func1: Functional, func2: Functional):
        super().__init__(func1, func2)
        self.func1.set_names_for_args(func2.arg_names())

    def __call__(
        self, *args, noise: Optional[Union[np.ndarray, float]] = None, **kwargs
    ):
        noise, args = self.prep_call_input(*args, noise=noise, **kwargs)
        return self.func1(
            self.func2(*args, noise=noise, **kwargs), noise=None, **kwargs
        )

    def function_str(self, variable_names: Collection[str] = None):
        rep_strs = ["", self.func2.function_str(variable_names=variable_names)]
        selected = int(self.inner_noise)

        if self.accepts_noise:
            rep_strs[selected] += f"{self.noise_transformation}(N)"
            if self.additive_noise:
                rep_strs[selected] += " + "
            else:
                rep_strs[selected] += " * "

        rep = rep_strs[0] + self.func1.function_str(variable_names=[rep_strs[1]])
        return rep


class BinaryFunctionalFromOp(BinaryFunctional, ABC):
    def __init__(self, func1: Functional, func2: Functional):
        super().__init__(func1, func2)
        self.position_to_func: Dict[int, Tuple[bool, bool]] = dict()
        self._assign_positions(func1, func2)

    @Functional.call_arg_handler
    def __call__(
        self, *args, noise: Optional[Union[np.ndarray, float]] = None, **kwargs
    ):
        left_args = []
        right_args = []
        for pos, arg in enumerate(args):
            left_yes, right_yes = self.position_to_func[pos]
            if left_yes:
                left_args.append(arg)
            if right_yes:
                right_args.append(arg)

        return self.operation(
            self.func1(*left_args, noise=noise, **kwargs),
            self.func2(*right_args, noise=noise, **kwargs),
        )

    def _assign_positions(self, func1, func2):
        """
        Assign positional args positions to the functionals.
        """
        f1_args = func1.arg_names()
        f2_args = func2.arg_names()
        for var, pos in self.args_to_position.items():
            self.position_to_func[pos] = var in f1_args, var in f2_args

    @abstractmethod
    def operation(
        self,
        evaluated_func1: Union[float, np.ndarray],
        evaluated_func2: Union[float, np.ndarray],
    ):
        raise NotImplementedError(
            "Functionals from binary operations must implement method `operation`."
        )

    def _assign_names_to_func(self, var_names: Collection[str]):
        left_vars = []
        right_vars = []
        left_var_names = self.func1.arg_names()
        right_var_names = self.func2.arg_names()
        for var in var_names:
            if var in left_var_names:
                left_vars.append(var)
            if var in right_var_names:
                right_vars.append(var)
        return left_vars, right_vars


class AddFunctional(BinaryFunctionalFromOp):
    def __init__(self, functional1: Functional, functional2: Functional):
        super().__init__(functional1, functional2)

    def operation(
        self,
        evaluated_func1: Union[float, np.ndarray],
        evaluated_func2: Union[float, np.ndarray],
    ):
        return evaluated_func1 + evaluated_func2

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        left, right = self._assign_names_to_func(variable_names)
        return self.func1.function_str(left) + " + " + self.func2.function_str(right)


class SubFunctional(BinaryFunctionalFromOp):
    def __init__(self, functional1: Functional, functional2: Functional):
        super().__init__(functional1, functional2)

    def operation(
        self,
        evaluated_func1: Union[float, np.ndarray],
        evaluated_func2: Union[float, np.ndarray],
    ):
        return evaluated_func1 - evaluated_func2

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        left, right = self._assign_names_to_func(variable_names)
        return self.func1.function_str(left) + " - " + self.func2.function_str(right)


class MulFunctional(BinaryFunctionalFromOp):
    def __init__(self, functional1: Functional, functional2: Functional):
        super().__init__(functional1, functional2)

    def operation(
        self,
        evaluated_func1: Union[float, np.ndarray],
        evaluated_func2: Union[float, np.ndarray],
    ):
        return evaluated_func1 * evaluated_func2

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        left, right = self._assign_names_to_func(variable_names)
        a1_string = self.func1.function_str(left)
        if self.func1.is_elemental:
            a1_string = "(" + a1_string + ")"

        a2_string = self.func2.function_str(right)
        if self.func2.is_elemental:
            a2_string = "(" + a2_string + ")"

        return a1_string + " * " + a2_string


class DivFunctional(BinaryFunctionalFromOp):
    def __init__(self, functional1: Functional, functional2: Functional):
        super().__init__(functional1, functional2)

    def operation(
        self,
        evaluated_func1: Union[float, np.ndarray],
        evaluated_func2: Union[float, np.ndarray],
    ):
        return evaluated_func1 / evaluated_func2

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        left, right = self._assign_names_to_func(variable_names)
        a1_string = self.func1.function_str(left)
        if self.func1.is_elemental:
            a1_string = "(" + a1_string + ")"

        a2_string = self.func2.function_str(right)
        if self.func2.is_elemental:
            a2_string = "(" + a2_string + ")"

        return a1_string + " / " + a2_string
