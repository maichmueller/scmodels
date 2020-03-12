import functools

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, TypeVar, Optional, List, Collection, Dict, Hashable, Callable


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
        vars: Collection[Hashable],
        allow_noise: bool = True,
        additive_noise: bool = True,
        is_argument_noise: bool = False,
        noise_transformation: Callable = lambda x: x,
        is_elemental: bool = True,
    ):
        self.allow_noise = allow_noise
        # whether we add the noise to the func or multiply it.
        self.additive_noise = additive_noise
        # whether the noise acts on the functionals' arguments or the functional itself ( f(N*x) vs N*f(x) )
        self.inner_noise = is_argument_noise
        # how the noise is treated inside the functional determined by a callable, that will be used when
        # noise is passed later on: N = self.noise_transformation(noise_array)
        if not allow_noise:
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
        self.args_to_position: Dict[Hashable, int] = dict()
        self.set_names_for_args(vars)
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

    @abstractmethod
    def __len__(self):
        """
        Method to return the number of variables the functional takes.
        """
        raise NotImplementedError(
            "Functional subclasses must provide '__len__' method."
        )

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
    def function_str(self, variable_names: Optional[Collection[Hashable]] = None):
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

    def str(self, variable_names: Optional[Collection[Hashable]] = None):
        if variable_names is None:
            if self.has_named_args:
                variable_names = list(self.args_to_position.keys())
            else:
                variable_names = [f"x_{i}" for i in range(len(self))]
        variable_names = ["N"] + variable_names
        functional = self.function_str(variable_names)
        prefix = f"f({', '.join(variable_names)}) = "
        return prefix + functional

    def set_names_for_args(self, names_collection: Collection[Hashable]):
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

    def get_arg_names(self):
        return set(self.args_to_position.keys())

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

            args = tuple(
                val
                for key, val in sorted(
                    kwargs.items(),
                    key=lambda key_val_pair: self.args_to_position[key_val_pair[0]],
                )
            )

        if self.allow_noise:
            noise = self.noise_transformation(noise)
            if self.inner_noise:
                if self.additive_noise:
                    args = (noise + args[0],)
                else:
                    args = (noise * args[0],)
                noise = None

        return noise, args

    def sprinkle_noise(
        self, output: Union[np.ndarray, float], noise: Union[np.ndarray, float]
    ):
        if not self.inner_noise:
            if self.additive_noise:
                output += noise
            else:
                output *= noise
        return output


class AddFunctional(Functional):
    def __init__(self, functional1: Functional, functional2: Functional):
        self.func1 = functional1
        self.func2 = functional2
        v_names = self.func1.get_arg_names() & (
            self.func2.get_arg_names()
        )  # & = set intersection
        super().__init__(vars=v_names, is_elemental=False)

    def __call__(self, noise, *args, **kwargs):
        return self.func1(noise, *args, **kwargs) + self.func2(noise, *args, **kwargs)

    def __len__(self):
        if self.func1.has_named_args and self.func2.has_named_args:
            return len(self.get_arg_names())

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return (
            self.func1.function_str(variable_names)
            + " + "
            + self.func2.function_str(variable_names)
        )


class SubFunctional(Functional):
    def __init__(self, functional1: Functional, functional2: Functional):
        self.assign1 = functional1
        self.assign2 = functional2
        super().__init__(is_elemental=False)

    def __call__(self, noise, *args, **kwargs):
        return self.assign1(noise, *args, **kwargs) - self.assign2(
            noise, *args, **kwargs
        )

    def __len__(self):
        if self.assign1.has_named_args and self.assign2.has_named_args:
            return len(
                self.assign1.get_arg_names().intersection(self.assign2.get_arg_names())
            )
        else:
            return len(self.assign1) + len(self.assign2)

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return (
            self.assign1.function_str(variable_names)
            + " - "
            + self.assign2.function_str(variable_names)
        )


class MulFunctional(Functional):
    def __init__(self, functional1: Functional, functional2: Functional):
        self.assign1 = functional1
        self.assign2 = functional2
        super().__init__(is_elemental=False)

    def __call__(self, noise, *args, **kwargs):
        return self.assign1(noise, *args, **kwargs) * self.assign2(
            noise, *args, **kwargs
        )

    def __len__(self):
        if self.assign1.has_named_args and self.assign2.has_named_args:
            return len(
                self.assign1.get_arg_names().intersection(self.assign2.get_arg_names())
            )
        else:
            return len(self.assign1) + len(self.assign2)

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        a1_string = self.assign1.function_str(variable_names)
        if self.assign1.is_elemental:
            a1_string = "(" + a1_string + ")"

        a2_string = self.assign1.function_str(variable_names)
        if self.assign2.is_elemental:
            a2_string = "(" + a2_string + ")"

        return a1_string + " * " + a2_string


class DivFunctional(Functional):
    def __init__(self, functional1: Functional, functional2: Functional):
        self.assign1 = functional1
        self.assign2 = functional2
        super().__init__(is_elemental=False)

    def __call__(self, noise, *args, **kwargs):
        return self.assign1(noise, *args, **kwargs) / self.assign2(
            noise, *args, **kwargs
        )

    def __len__(self):
        if self.assign1.has_named_args and self.assign2.has_named_args:
            return len(
                self.assign1.get_arg_names().intersection(self.assign2.get_arg_names())
            )
        else:
            return len(self.assign1) + len(self.assign2)

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        a1_string = self.assign1.function_str(variable_names)
        if self.assign1.is_elemental:
            a1_string = "(" + a1_string + ")"

        a2_string = self.assign1.function_str(variable_names)
        if self.assign2.is_elemental:
            a2_string = "(" + a2_string + ")"

        return a1_string + " / " + a2_string


class CompositionFunctional(Functional):
    r"""
    The Linear Functional function of the form:
        f(X_S, N) = offset + \sum_{i \in S} a_i * X_i + noise_coeff * N
    """

    def __init__(self, outer: Functional, inner: Functional):
        self.func_outer = outer
        self.func_inner = inner
        super().__init__(is_elemental=False)

    def __call__(self, noise: Union[float, np.ndarray], *args, **kwargs):
        args = self.prep_call_input(*args, **kwargs)
        return self.func_outer(noise, self.func_inner(noise, *args, **kwargs), **kwargs)

    def __len__(self):
        return len(self.func_inner)

    def function_str(self, variable_names: Collection[Hashable] = None):
        rep_strs = ["", self.func_inner.function_str(variable_names=variable_names)]
        selected = int(self.inner_noise)

        if self.allow_noise:
            rep_strs[selected] += f"{self.noise_transformation}(N)"
            if self.additive_noise:
                rep_strs[selected] += " + "
            else:
                rep_strs[selected] += " * "

        rep = rep_strs[0] + self.func_outer.function_str(variable_names=[rep_strs[1]])
        return rep
