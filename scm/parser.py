from collections import deque, defaultdict
import re as regex

import sympy
from sympy.functions import *
from sympy.stats import *
from sympy.stats import __all__ as all_stats_imports

from typing import (
    List,
    Union,
    Dict,
    Tuple,
    Iterable,
    Set,
    Mapping,
    Sequence,
    Collection,
    Optional,
    Hashable,
    Sequence,
    Type,
)

all_stats_imports = set(all_stats_imports)


def parse_assignments(assignment_strs: List[str]):
    var_p = regex.compile(r"(?<=([(]|[[)*+-/%]))\w+(?=([)*+-/%]+|$))")
    digit_p = regex.compile(r"^\d+$")

    functional_map = dict()
    for assignment in assignment_strs:
        assign_split = assignment.split("=", 1)
        parents, assignment_list = [], assign_split[1].split(",", 1)[0]
        avar, astr = tuple(map(lambda s: "".join(s.split()), assign_split))
        afunc, anoise = astr.split(",", 1)
        noise_var, model_sym = allocate_noise_model(anoise)
        for match_obj in var_p.finditer(astr):
            matched_str = match_obj.group()
            if digit_p.search(matched_str) is not None:
                # exclude digit only matches (these aren't variable names)
                continue
            else:
                # the matched str is considered a full variable name
                if not matched_str == noise_var:
                    parents.append(matched_str)
        functional_map[avar] = parents, assignment_list.strip(), model_sym
    return functional_map


def allocate_noise_model(noise_assignment: str):
    noise_var, model = noise_assignment.split("~")
    par_idx = model.find("(") + 1
    if model[:par_idx-1] not in all_stats_imports:
        # crude check whether the noise model is
        raise ValueError(f"noise model {model[:par_idx-1]} not supported.")
    model = model[:par_idx] + r'"' + noise_var + r'",' + model[par_idx:]
    model_sym = []
    exec(f"model_sym.append({model})")
    return noise_var, model_sym[0]
