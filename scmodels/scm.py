from scmodels.parser import parse_assignments, extract_parents

import random
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from collections import deque, defaultdict
from itertools import repeat

import matplotlib.pyplot as plt
import logging
from copy import deepcopy

import sympy
from sympy.core.singleton import SingletonRegistry
from sympy.functions import *
from sympy.stats import sample
from sympy.stats.rv import RandomSymbol

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
    Generator,
    Iterator,
    Callable,
)

RV = RandomSymbol
AssignmentMap = Dict[str, Tuple[str, RV]]
FunctionalMap = Dict[str, Tuple[List[str], Callable, RV]]


class Assignment:

    noise_argname = "{__noise__}"

    def __init__(
        self,
        functor: Callable,
        ordered_parents: Sequence[str],
        desc: Optional[str] = None,
    ):
        assert callable(functor), "Passed functor must be callable."
        self.functor = functor
        self.desc = desc if desc is not None else "__custom__"
        self.arg_positions = {
            var: pos for (var, pos) in zip(ordered_parents, range(len(ordered_parents)))
        }

    def __len__(self):
        return len(self.arg_positions)

    def __call__(self, *args, **kwargs):
        args = list(args) + [None] * (len(self) - len(args))
        for var in kwargs:
            if var in self.arg_positions:
                args[self.arg_positions[var]] = kwargs[var]
        return self.functor(*args)

    def __str__(self):
        return self.desc


class SCM:
    """
    Class building a Structural Causal Model.

    With this class one can sample causally from the underlying graph and also perform do-interventions and
    soft-interventions, as well as more general interventions targeting one or more variables with arbitrary
    changes to the assignment and/or noise structure (including parent-child relationship changes).

    For visualization aid, the SCM can plot itself and also print a summary of itself to the console.
    To this end, the decision was made to limit the potential input to the SCM objects in terms of
    functional and noise functors.
    If a user wishes to add a custom functional or noise function to their usage of the SCM, they will need
    to provide implementations inheriting from the respective base classes of noise or functional.

    Notes
    -----
    When constructing or intervening the user is responsible to guarantee acyclicity in the SCM. No checks are enabled
    at this stage.
    """

    def __init__(
        self,
        assignments: Union[AssignmentMap, FunctionalMap, Sequence[str]],
        variable_tex_names: Optional[Dict] = None,
        seed: Optional[int] = None,
        scm_name: str = "Structural Causal Model",
    ):
        """
        Construct the SCM from an assignment map with the variables as keys and its assignment information
        as tuple of parents, assignment, and noise distribution or provide the assignments in a list of strings, which
        directly tell

        Notes
        -----
        Note, that the assignment string needs to align with the parents list namewise!

        Examples
        --------
        To set up 3 variables of the form

        .. math:: X_0 = LogListic(1, 1)
        .. math:: X_1 = 3 (X_0)^2 + Normal(1, 0.5)
        .. math::   Y = 2 * X_0 - sqrt(X_1) + Normal(2, 1)

        we can either build an assignment map

        >>> from sympy.stats import LogLogistic, LogNormal
        ...
        ... assignment = {
        ...     "X_zero": (
        ...         "N",
        ...         LogLogistic("N", alpha=1, beta=1)
        ...     ),
        ...     "X_1": (
        ...         "N * 3 * X_zero ** 2",
        ...         LogNormal("N", mean=1, std=1),
        ...     "Y": (
        ...         "N + 2 * X_zero + sqrt(X_1)",
        ...         Normal("N", mean=2, std=1),
        ...     ),
        ... }

        to initialize the scm with it

        >>> causal_net = SCM(
        ...     assignments=assignment,
        ...     variable_tex_names={"X_zero": "$X_{zero}$", "X_1": "$X_1$"}
        ... )

        or we can build the SCM directly from assignment strings

        >>> causal_net = SCM(
        ...     [
        ...         "X_zero = N, N ~ LogLogistic(alpha=1, beta=1)",
        ...         "X_1 = N * 3 * X_zero ** 2, N ~ LogNormal(mean=1, std=1)",
        ...         "Y = N + 2 * X_zero + sqrt(X_1), N ~ Normal(mean=2, std=1)"
        ...     ],
        ...     variable_tex_names={"X_zero": "$X_{zero}$", "X_1": "$X_1$"}
        ... )

        Parameters
        ----------
        assignments: dict,
            The functional dictionary for the SCM construction as explained above.
        variable_tex_names: (optional) Dict,
            A collection of the latex names for the variables in the causal graph. The dict needs to provide a tex name
            for each passed variable name in the form: {"VarName": "VarName_in_TeX"}.
            Any variable that is missing in the dictionary will be assumed to accept its current name as the TeX
            version.
            If not provided defaults to the input names in the functional map.
        seed: (optional) str,
            Seeding the graph for reproducibility.
        scm_name: (optional) str,
            The name of the SCM. Default is 'Structural Causal Model'.
        """
        self.scm_name: str = scm_name
        self.rng_state = np.random.default_rng()  # we always have a local RNG machine
        self.seed(seed)
        self.nr_variables: int = len(assignments)

        # the attribute list that any given node in the graph has.
        (
            self.assignment_key,
            self.noise_key,
            self.noise_repr_key
        ) = ("assignment", "noise", "noise_repr")

        # a backup dictionary of the original assignments of the intervened variables,
        # in order to undo the interventions later.
        self.interventions_backup_attr: Dict = dict()
        self.interventions_backup_parent: Dict = dict()

        # build the graph:
        # any node will be given the attributes of function and noise to later sample from and also an incoming edge
        # from its causal parent. We will store the causal root nodes separately.
        self.dag = nx.DiGraph()
        self.roots: List = []
        self.insert(assignments)

        self.var = np.array(self.get_variables())
        # supply any variable name, that has not been assigned a different TeX name, with itself as TeX name.
        # This prevents missing labels in the plot method.
        if variable_tex_names is not None:
            for name in self.var:
                if name not in variable_tex_names:
                    variable_tex_names[name] = name
            # the variable names as they can be used by the plot function to draw the names in TeX mode.
            self.var_draw_dict: Dict = variable_tex_names
        else:
            self.var_draw_dict = {name: name for name in self.var}

    def __getitem__(self, node):
        return self.dag.pred[node], self.dag.nodes[node]

    def __str__(self):
        return self.str()

    def sample(
        self,
        n: int,
        variables: Optional[Sequence[Hashable]] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        """
        Sample method to generate data for the given variables. If no list of variables is supplied, the method will
        simply generate data for all variables.
        Setting the seed guarantees reproducibility.

        Parameters
        ----------
        n: int,
            number of samples
        variables: list,
            the variable names to consider for sampling. If None, all variables will be sampled.
        seed: int,
            the seeding for the noise generators

        Returns
        -------
        pd.DataFrame,
            the dataframe containing the samples of all the variables needed for the selection.
        """

        samples = dict()
        if seed is None:
            seed = self.rng_state

        for node in self._causal_iterator(variables):
            node_attr = self.dag.nodes[node]
            predecessors = list(self.dag.predecessors(node))

            named_args = dict()
            for pred in predecessors:
                named_args[pred] = samples[pred]

            noise_gen = node_attr[self.noise_key]
            if noise_gen is not None:
                named_args[Assignment.noise_argname] = np.array(
                    list(sample(noise_gen, numsamples=n, seed=seed)), dtype=float
                )

            data = node_attr[self.assignment_key](**named_args)
            samples[node] = data
        return pd.DataFrame.from_dict(samples)

    def sample_iter(
        self,
        container: Optional[Dict[str, List]] = None,
        variables: Optional[Sequence[Hashable]] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> Iterator[Dict[str, List]]:
        """
        Sample method to generate data for the given variables. If no list of variables is supplied, the method will
        simply generate data for all variables.
        Setting the seed guarantees reproducibility.

        Parameters
        ----------
        container: Dict[str, List] (optional),
            the container in which to store the output. If none is provided, the sample is returned in a new container,
            otherwise the provided container is filled with the samples and returned.
        variables: list,
            the variable names to consider for sampling. If None, all variables will be sampled.
        seed: int or np.random.Generator,
            the seeding for the noise generators

        Returns
        -------
        pd.DataFrame,
            the dataframe containing the samples of all the variables needed for the selection.
        """
        if seed is None:
            seed = self.rng_state
        vars_ordered = list(self._causal_iterator(variables))
        noise_iters = []
        for node in vars_ordered:
            noise_gen = self.dag.nodes[node][self.noise_key]
            if noise_gen is not None:
                noise_iters.append(
                    sample(noise_gen, numsamples=SingletonRegistry.Infinity, seed=seed)
                )
            else:
                noise_iters.append(repeat(None))

        if container is None:
            samples = {var: [] for var in vars_ordered}
        else:
            samples = container
        while True:
            for i, node in enumerate(vars_ordered):
                node_attr = self.dag.nodes[node]
                predecessors = list(self.dag.predecessors(node))

                named_args = dict()
                for pred in predecessors:
                    named_args[pred] = samples[pred][-1]

                noise = next(noise_iters[i])
                if noise is not None:
                    named_args[Assignment.noise_argname] = noise

                data = node_attr[self.assignment_key](**named_args)
                samples[node].append(data)
            yield samples

    def intervention(
        self, interventions: Dict[str, Tuple[Optional[str], Optional[RV]]],
    ):
        """
        Method to apply interventions on the specified variables.

        One can set any variable to a new functional function and noise model, thus redefining its parents and their
        dependency structure. Using this method will enable the user to call sample, plot etc. on the SCM just like
        before.
        In order to allow for undoing the intervention(s), the original state of the variables in the network is saved
        as backup and can be undone by calling ``undo_interventions``.

        Parameters
        ----------
        interventions: dict,
            the variables as keys and their new functional as values. For the values one can
            choose between a dictionary or a list-like.
            - For dict: the dictionary is assumed to have the optional keys (default behaviour explained):
                -- "function_key": str, the new assignment as str. If None, keeps current assignment.
                -- "noise_key": sympy.rv, the new noise RV. If None, keeps current noise model.

            - For sequence: the order is (assignment str, noise RV).
                In order to omit one of these, set them to None.
        """
        interventional_map = dict()
        for var, items in interventions.items():
            if var not in self.get_variables():
                logging.warning(f"Variable '{var}' not found in graph. Omitting it.")
                continue

            if isinstance(items, dict):
                if any(
                    (
                        key not in (self.assignment_key, self.noise_key)
                        for key in items.keys()
                    )
                ):
                    raise ValueError(
                        f"Intervention dictionary provided with the wrong keys.\n"
                        f"Observed keys are: {list(items.keys())}\n"
                        f"Possible keys are: ['{self.assignment_key}', '{self.noise_key}']"
                    )

                attr_dict = items

            elif isinstance(items, Sequence):
                items = [elem for elem in items]
                assert (
                    len(items) == 2
                ), "The positional items container needs to contain exactly 2 items."

                existing_attr = self[var][1]
                if items[0] is None:
                    items[0] = existing_attr[self.assignment_key]
                if items[1] is None:
                    items[1] = existing_attr[self.noise_key]
                attr_dict = {
                    self.assignment_key: items[0],
                    self.noise_key: items[1],
                }

            else:
                raise ValueError(
                    f"Intervention items container '{type(items)}' not supported."
                )

            if var not in self.interventions_backup_attr:
                # the variable has NOT already been backed up. If we overwrite it now it is fine. If it had already been
                # in the backup, then it we would need to pass on this step, in order to not overwrite the backup
                # (possibly with a previous intervention)
                self.interventions_backup_attr[var] = deepcopy(self[var][1])

            if var not in self.interventions_backup_parent:
                # the same logic goes for the parents backup.
                parent_backup = list(self.dag.predecessors(var))
                self.interventions_backup_parent[var] = parent_backup
                self.dag.remove_edges_from([(parent, var) for parent in parent_backup])
            # patch up the attr dict as the provided items were merely strings and now need to be parsed by sympy.
            interventional_map[var] = (
                attr_dict[self.assignment_key],
                attr_dict[self.noise_key],
            )
        self.insert(interventional_map)

    def do_intervention(self, var_val_pairs: Sequence[Tuple[str, float]]):
        """
        Perform do-interventions, i.e. setting specific variables to a constant value.
        This method removes the noise influence of the intervened variables.

        Convenience wrapper around ``interventions`` method.

        Parameters
        ----------
        var_val_pairs Sequence of str-float tuple,
            the variables to intervene on with their new values.
        """
        interventions_dict = dict()
        for var, val in var_val_pairs:
            interventions_dict[var] = (str(val), None)
        self.intervention(interventions_dict)

    def soft_intervention(
        self, variables: Sequence[str], noise_models: Sequence[RV],
    ):
        """
        Perform noise interventions, i.e. modifying the noise generator of specific variables.

        Convenience wrapper around ``interventions`` method.

        Parameters
        ----------
        variables : Sequence,
            the variables to intervene on.
        noise_models : Sequence[float],
            the constant values the chosen variables should be set to.
        """
        if len(variables) != len(noise_models):
            raise ValueError(
                f"Got {len(variables)} variables, but {len(noise_models)} noise models."
            )

        interventions_dict = dict()
        for var, noise in zip(variables, noise_models):
            interventions_dict[var] = (None, noise)
        self.intervention(interventions_dict)

    def undo_intervention(self, variables: Optional[Sequence[str]] = None):
        """
        Method to undo previously done interventions.

        The variables whose interventions should be made undone can be provided in the ``variables`` argument. If no
        list is supplied, all interventions will be undone.

        Parameters
        ----------
        variables: list-like,
            the variables to be undone.
        """
        if variables is not None:
            present_variables = self.filter_variable_names(variables, self.dag)
        else:
            present_variables = list(self.interventions_backup_attr.keys())

        for var in present_variables:
            try:
                attr_dict = self.interventions_backup_attr.pop(var)
                parents = self.interventions_backup_parent.pop(var)
            except KeyError:
                logging.warning(
                    f"Variable '{var}' not found in intervention backup. Omitting it."
                )
                continue

            self.dag.add_node(var, **attr_dict)
            self.dag.remove_edges_from(
                [(parent, var) for parent in self.dag.predecessors(var)]
            )
            self.dag.add_edges_from([(parent, var) for parent in parents])

    def d_separated(
        self, x: Sequence[str], y: Sequence[str], s: Optional[Sequence[str]] = None
    ):
        """
        Checks if all variables in X are d-separated from all variables in Y by the variables in S.

        Parameters
        ----------
        x: Sequence,
            First set of nodes in DAG.

        y: Sequence,
            Second set of nodes in DAG.

        s: Sequence (optional),
            Set of conditioning nodes in DAG.

        Returns
        -------
        bool,
            A boolean indicating whether x is d-separated from y by s.
        """
        return nx.d_separated(
            self.dag, set(x), set(y), set(s) if s is not None else set()
        )

    def is_dag(self):
        """

        Returns
        -------
        bool,
            A boolean indicating whether the current graphical model is indeed a DAG.
        """
        return nx.is_directed_acyclic_graph(self.dag)

    def insert(self, assignments: Union[Sequence[str], AssignmentMap]):
        """
        Method to insert variables into the graph. The assignments can be passed either as full string assignments or
        via an assignment map.

        Parameters
        ----------
        assignments, Sequence of str or dictionary of nodes to assignment tuples
            The assignments to add to the graph.

        """
        if isinstance(assignments, Sequence):
            assignments = parse_assignments(assignments)
        elif not isinstance(assignments, Dict):
            raise ValueError(
                f"Assignments parameter accepts either be a "
                f"Sequence[str], "
                f"a {str(AssignmentMap).replace('typing.', '')}, "
                f"or a {str(FunctionalMap).replace('typing.', '')}"
            )

        for (node_name, assignment_pack) in assignments.items():
            if len(assignment_pack) == 2:
                assignment_str, noise_model = assignment_pack
                parents = extract_parents(assignment_str, noise_model)

                noise, assignment_func = sympify_assignment(
                    parents, assignment_str, noise_model
                )

            elif len(assignment_pack) == 3:
                parents, assignment_func, noise_model = assignment_pack
                assert callable(
                    assignment_func
                ), "Assignment tuple holds 3 elements, but the functional entry is not callable."
                assignment_str = None
            else:
                raise ValueError("Assignment entry must be a tuple of 2 or 3 entries.")

            if len(parents) > 0:
                for parent in parents:
                    self.dag.add_edge(parent, node_name)
            else:
                self.roots.append(node_name)

            if noise_model is not None:
                parents = [Assignment.noise_argname] + list(parents)

            attr_dict = {
                self.assignment_key: Assignment(assignment_func, parents, assignment_str),
                self.noise_key: noise_model,
                self.noise_repr_key: extract_rv_desc(noise_model),
            }

            self.dag.add_node(
                node_name, **attr_dict,
            )

    def seed(
        self, seed: Optional[Union[int, np.random.Generator, np.random.RandomState]]
    ):
        """
        Seeds the assignments.

        Parameters
        ----------
        seed: int,
            The seed to use for rng.
        """
        if seed is not None:
            if isinstance(seed, int):
                self.rng_state = np.random.default_rng(seed=seed)
            elif isinstance(seed, (np.random.Generator, np.random.RandomState)):
                self.rng_state = seed
            else:
                raise ValueError(f"seed type {type(seed)} not supported.")

    def plot(
        self,
        draw_labels: bool = True,
        node_size: int = 500,
        figsize: Tuple[int, int] = (6, 4),
        dpi: int = 150,
        alpha: float = 0.5,
        savepath: Optional[str] = None,
        **kwargs,
    ):
        """
        Plot the causal graph of the bayesian_graphs in a dependency oriented way.

        This will attempt a tree plot of the bayesian_graphs, in the case that the graph is indeed a tree.
        However, because a causal graph is a DAG and can thus have directionless cycles (but not directional cycles), a
        tree structure often can't be computed. Therefore this method relies on graphviz to compute a feasible
        representation of the causal graph.

        The graphviz package has been marked as an optional package for this module and therefore needs to be installed
        by the user.
        Note, that graphviz may demand further libraries to be supplied, thus the following
        command should install the necessary dependencies on Ubuntu, if graphviz couldn't be found on your system.
        Open a terminal and type:

            ``sudo apt-get install graphviz libgraphviz-dev pkg-config``

        Notes
        -----
        One will need to call ``plt.show()`` for the plot to be printed to the backend.

        Parameters
        ----------
        draw_labels : (optional) bool,
            Whether to draw the node labels onto the node. Can look unwieldy if the names are long.
            Default is True.
        node_size : (optional) int,
            the size of the node circles in the graph. Bigger values mean bigger circles.
            Default is 500.
        figsize : (optional) tuple,
            the size of the figure to be passed to matplotlib. Default is (6, 4).
        dpi : (optional) int,
            the dots per inch arg for matplotlib. Default is 150.
        alpha : (optional) float,
            the statistical significance level for the test. Default value is 0.05.
        savepath: (optional) str,
            the full filepath to the, to which the plot should be saved. If not provided, the plot will not be saved.
        kwargs :
            arguments to be passed to the ``networkx.draw`` method. Check its documentation for a full list.

        Returns
        -------
        tuple,
            the plt.figure and figure-axis objects holding the graph plot.
        """
        if nx.is_tree(self.dag):
            pos = hierarchy_pos(self.dag, root=self.roots[0])
        else:
            pos = graphviz_layout(self.dag, prog="dot")
        plt.title(self.scm_name)
        if draw_labels:
            labels = self.var_draw_dict
        else:
            labels = {}
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        nx.draw(
            self.dag,
            pos=pos,
            ax=ax,
            labels=labels,
            with_labels=True,
            node_size=node_size,
            alpha=alpha,
            **kwargs,
        )
        if savepath is not None:
            fig.savefig(savepath)
        return fig, ax

    def str(self):
        """
        Computes a string representation of the SCM. Specifically, returns a string of the form:

        'Structural Causal Model of `nr_variables` variables: X_0, X_1, ...
         Following variables are actively intervened on: [`list(intervened variables)`]
         Current Functional Functions are:
         X_0 := f(N) = `X_0 functional_string_representation`
         X_1 := f(N, X_0) = `X_1 functional_string_representation`
         ...'

        Returns
        -------
        str,
            the representation.
        """
        lines = [
            f"Structural Causal Model of {self.nr_variables} variables: "
            + ", ".join(self.var),
            f"Following variables are actively intervened on: {list(self.interventions_backup_attr.keys())}",
            "Current Assignments are:",
        ]
        max_var_space = max([len(var_name) for var_name in self.var])
        for node in self.dag.nodes:
            attr_dict = self[node][1]
            noise_symbol = str(attr_dict[self.noise_key])
            parents_var = [noise_symbol] + [
                pred for pred in self.dag.predecessors(node)
            ]
            args_str = ", ".join(parents_var)
            line = f"{str(node).rjust(max_var_space)} := f({args_str}) = {attr_dict[self.assignment_repr_key]}"
            # add explanation to the noise term
            if noise_symbol is not None:
                line += f"\t [ {noise_symbol} ~ {str(attr_dict[self.noise_repr_key])} ]"
            lines.append(line)
        return "\n".join(lines)

    def get_variables(self, causal_order=True):
        if causal_order:
            return list(self._causal_iterator())
        else:
            return list(self.dag.nodes)

    @staticmethod
    def filter_variable_names(variables: Iterable, dag: nx.DiGraph):
        """
        Filter out variable names, that are not currently in the graph. Warn for each variable that wasn't present.

        Returns a generator which iterates over all variables that have been found in the graph.

        Parameters
        ----------
        variables: list,
            the variables to be filtered

        dag: nx.DiGraph,
            the DAG in which the variables are supposed to be found.

        Returns
        -------
        generator,
            generates the filtered variables in sequence.
        """
        for variable in variables:
            if variable in dag.nodes:
                yield variable
            else:
                logging.warning(
                    f"Variable '{variable}' not found in graph. Omitting it."
                )

    def _causal_iterator(
        self, variables: Optional[Iterable] = None, dag: Optional[nx.DiGraph] = None
    ):
        """
        Provide a causal iterator through the graph starting from the roots going to the variables needed.

        This iterator passes only the ancestors of the variables and thus is helpful in filtering out all the variables
        that have no causal effect on the desired variables.

        Parameters
        ----------
        variables: list,
            the names of all the variables that are to be considered. Names that cannot be found in the naming list of
            the graph will be ignored (warning raised).

        Yields
        ------
        Hashable,
            the node object used to denote nodes in the graph in causal order. These are usually str or ints, but can be
            any hashable type passable to a dict.
        """
        if dag is None:
            dag = self.dag
        if variables is None:
            for node in nx.topological_sort(dag):
                yield node
            return
        visited_nodes: Set = set()
        var_causal_priority: Dict = defaultdict(int)
        queue = deque([var for var in self.filter_variable_names(variables, dag)])
        while queue:
            nn = queue.popleft()
            # this line appears to be pointless, but is necessary to emplace the node 'nn' in the dict with its current
            # value, if already present, otherwise with the default value (0).
            var_causal_priority[nn] = var_causal_priority[nn]
            if nn not in visited_nodes:
                for parent in dag.predecessors(nn):
                    var_causal_priority[parent] = max(
                        var_causal_priority[parent], var_causal_priority[nn] + 1
                    )
                    queue.append(parent)
                visited_nodes.add(nn)
        for key, _ in sorted(var_causal_priority.items(), key=lambda x: -x[1]):
            yield key


def sympify_assignment(parents: Iterable[str], assignment_str: str, noise_model: RV):
    """
    Parse the provided assignment string with sympy and then lambdifies it, to be used as a normal function.

    Parameters
    ----------
    assignment_str: str, the assignment to parse.
    parents: Iterable, the parents' names.
    noise_model: RV, the random variable inside the assignment.

    Returns
    -------
    function,
        the lambdified assignment.
    """

    symbols = []
    if noise_model is not None:
        symbols.append(noise_model)
        # let the noise models variable name be known as symbol. This is necessary for sympifying.
        exec(f"{str(noise_model)} = noise_model")
    for par in parents:
        exec(f"{par} = sympy.Symbol('{par}')")
        symbols.append(eval(par))
    assignment = sympy.sympify(eval(assignment_str))
    try:
        assignment = sympy.lambdify(symbols, assignment, "numpy")
    except NameError as e:
        warnings.warn(
            f"The assignment string could not be resolved in numpy, the error message reads: {e}\n"
            f"Lambdifying without numpy.",
        )
        assignment = sympy.lambdify(symbols, assignment)
    return noise_model, assignment


def extract_rv_desc(rv: Optional[RV]):
    """
    Extracts a human readable string description of the random variable.

    Parameters
    ----------
    rv: RV,
        the random variable.

    Returns
    -------
    str, the description.
    """
    if rv is None:
        return str(None)

    dist = rv.pspace.args[1]
    argnames = dist._argnames
    args = dist.args
    dist_name = str(dist).split("(", 1)[0].replace("Distribution", "")
    full_rv_str = f"{dist_name}({', '.join([f'{arg}={val}' for arg, val in zip(argnames, args)])})"
    return full_rv_str


def hierarchy_pos(
    graph: nx.Graph,
    root=None,
    width=1.0,
    vert_gap=0.2,
    vert_loc=0,
    leaf_vs_root_factor=0.5,
    check_for_tree=True,
):
    """
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).

    There are two basic approaches we think of to allocate the horizontal
    location of a node.

    - Top down: we allocate horizontal space to a node.  Then its ``k``
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.

    We use use both of these approaches simultaneously with ``leaf_vs_root_factor``
    determining how much of the horizontal space is based on the bottom up
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.


    Parameters
    ----------
    **G** the graph (must be a tree)

    **root** the root node of the tree
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root

    **leaf_vs_root_factor**
    """
    if check_for_tree and not nx.is_tree(graph):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(graph, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(graph))
            )  # allows back compatibility with nx version 1.11
        else:
            root = np.random.choice(list(graph.nodes))

    def __hierarchy_pos(
        graph_,
        root_,
        leftmost_,
        width_,
        leaf_dx_=0.2,
        vert_gap_=0.2,
        vert_loc_=0,
        xcenter_=0.5,
        root_pos_=None,
        leaf_pos_=None,
        parent_=None,
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if root_pos_ is None:
            root_pos_ = {root_: (xcenter_, vert_loc_)}
        else:
            root_pos_[root_] = (xcenter_, vert_loc_)
        if leaf_pos_ is None:
            leaf_pos_ = {}
        children = list(graph_.neighbors(root_))
        leaf_count = 0
        if not isinstance(graph_, nx.DiGraph) and parent_ is not None:
            children.remove(parent_)
        if len(children) != 0:
            root_dx = width_ / len(children)
            nextx = xcenter_ - width_ / 2 - root_dx / 2
            for child in children:
                nextx += root_dx
                root_pos_, leaf_pos_, new_leaves = __hierarchy_pos(
                    graph_,
                    child,
                    leftmost_ + leaf_count * leaf_dx_,
                    width_=root_dx,
                    leaf_dx_=leaf_dx_,
                    vert_gap_=vert_gap_,
                    vert_loc_=vert_loc_ - vert_gap_,
                    xcenter_=nextx,
                    root_pos_=root_pos_,
                    leaf_pos_=leaf_pos_,
                    parent_=root_,
                )
                leaf_count += new_leaves

            leftmostchild = min(
                (x for x, y in [leaf_pos_[child] for child in children])
            )
            rightmostchild = max(
                (x for x, y in [leaf_pos_[child] for child in children])
            )
            leaf_pos_[root_] = ((leftmostchild + rightmostchild) / 2, vert_loc_)
        else:
            leaf_count = 1
            leaf_pos_[root_] = (leftmost_, vert_loc_)
        #        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
        print(leaf_count)
        return root_pos_, leaf_pos_, leaf_count

    xcenter = width / 2.0
    if isinstance(graph, nx.DiGraph):
        leaf_count = len(
            [
                node
                for node in nx.descendants(graph, root)
                if graph.out_degree(node) == 0
            ]
        )
    elif isinstance(graph, nx.Graph):
        leaf_count = len(
            [
                node
                for node in nx.node_connected_component(graph, root)
                if graph.degree(node) == 1 and node != root
            ]
        )
    else:
        raise ValueError(
            "Passed graph is neither a networkx.DiGraph nor networkx.Graph."
        )
    root_pos, leaf_pos, leaf_count = __hierarchy_pos(
        graph,
        root,
        0,
        width,
        leaf_dx_=width * 1.0 / leaf_count,
        vert_gap_=vert_gap,
        vert_loc_=vert_loc,
        xcenter_=xcenter,
    )
    pos = {}
    for node in root_pos:
        pos[node] = (
            leaf_vs_root_factor * leaf_pos[node][0]
            + (1 - leaf_vs_root_factor) * root_pos[node][0],
            leaf_pos[node][1],
        )
    xmax = max(x for x, y in pos.values())
    for node in pos:
        pos[node] = (pos[node][0] * width / xmax, pos[node][1])
    return pos
