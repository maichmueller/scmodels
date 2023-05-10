from collections import deque, defaultdict
import re as regex

import sympy
from sympy.functions import *
from sympy.stats import *
from sympy.stats.rv import RandomSymbol


from typing import *

all_stats_imports = set(sympy.stats.__all__)

var_p = regex.compile(r"(?<=([(]|[)*+-/%]))\w+(?=([)*+-/%]+|$))|^\w+(?=([)*+-/%]+|$))")
digit_p = regex.compile(r"^\d+$")


def parse_assignments(assignment_strs: Sequence[str]):
    """
    This parses a list of assignment strings. The assignments are supposed to be given in the following form:

    'NEW_VAR = f(Parent_1, ..., Parent_n, N), N ~ DISTRIBUTION()'

    Any element is supposed to be named after your present use case. The function f is whatever the
    assignment is meant to do, e.g. f(X, Y, N) = N + X * Y for additive noise and multiplied parents.
    These functions need to be parsable by sympy to be correct.

    Parameters
    ----------
    assignment_strs: list
        The assignment strings.

    Returns
    -------
    dict,
        The functional map of variables with their parents, assignment strings, and noise models as needed to construct
        an SCM object.
    """
    functional_map = dict()
    for assignment in assignment_strs:

        # split the assignment 'X = f(Parents, Noise), Noise ~ D' into [X, f(Parents, Noise), Noise ~ D]
        assign_var, assignment_n_noise = assignment.split("=", 1)
        assign_noise_split = assignment_n_noise.split(",", 1)

        if len(assign_noise_split) == 1:
            # this is the case when there was no ',' separating functional body and noise distribution specification
            assign_str = assign_noise_split[0]
            model_sym = []
        else:
            assign_str, noise_str = assign_noise_split
            _, model_sym = allocate_noise_model(strip_whitespaces(noise_str).split("/"))
        functional_map[assign_var.strip()] = assign_str.strip(), model_sym
    return functional_map


def extract_parents(assignment_str: str, noise_var: List[Union[str, sympy.Symbol]]) -> List[str]:
    """
    Extract the parent variables in an assignment string.

    Examples
    --------
    For the following assignment

    >>> 'N + sqrt(Z_0) * log(Y_2d)'

    this method should return the following

    >>> extract_parents('N + sqrt(Z_0) * log(Y_2d)', 'N')
    >>> ['Z_0', 'Y_2d']

    Parameters
    ----------
    assignment_str: str
        the assignment str (without '=' sign and noise distribution).

    noise_var: str or sympy symbol,
        the identifier of the noise variable (excluded from parents list)

    Returns
    -------
    list,
        the parents found in the string
    """
    noise_var = str(noise_var)
    # set does not preserve insertion order, so we need to bypass this issue with a list
    parents = []
    for match_obj in var_p.finditer(strip_whitespaces(assignment_str)):
        matched_str = match_obj.group()
        if digit_p.search(matched_str) is not None:
            # exclude digit only matches (these aren't variable names)
            continue
        else:
            # the matched str is considered a full variable name
            if matched_str not in noise_var:
                parents.append(matched_str)
    # the idea of the return value is to remove duplicates while preserving the insertion order.
    # 'set' would do the same as dict.fromkeys, however, 'set' discards the order,
    # while dict.fromkeys preserves order! That is why we cannot simply call set followed by list here,
    # without creating the wrong causal order in our list.
    # (This is also faster than list(np.unique(parents)), as per benchmark)
    return list(dict.fromkeys(parents))


def allocate_noise_model(noise_assignments: List[str]):
    model_symbs = [None] * len(noise_assignments)
    noise_vars = [None] * len(noise_assignments)
    for i, noise_assignment in enumerate(noise_assignments):
        noise_var, model = noise_assignment.split("~")
        noise_vars[i] = noise_var
        par_idx = model.find("(") + 1
        if model[:par_idx-1] not in all_stats_imports:
            # crude check whether the noise model is supported
            raise ValueError(f"noise model {model[:par_idx-1]} not supported. Check for spelling errors.")
        model = model[:par_idx] + r'"' + noise_var + r'",' + model[par_idx:]
        exec(f"model_symbs[i] = {model}")
    return noise_vars, model_symbs


def strip_whitespaces(s: str):
    return "".join(s.split())
