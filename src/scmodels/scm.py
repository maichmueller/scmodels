import logging
import warnings
from collections import deque, defaultdict
from copy import deepcopy
from itertools import repeat
from typing import (
    List,
    Union,
    Dict,
    Tuple,
    Iterable,
    Set,
    Hashable,
    Optional,
    Sequence,
    Iterator,
    Callable,
)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sympy
from networkx.drawing.nx_agraph import graphviz_layout
from sympy.core.singleton import SingletonRegistry
from sympy.stats import sample, sample_iter, random_symbols
from sympy.stats.rv import RandomSymbol
from sympy.functions import *

from scmodels.parser import parse_assignments, extract_parents

RV = RandomSymbol
AssignmentSeq = Sequence[str]
AssignmentMap = Dict[str, Union[Tuple[str, RV], Tuple[List[str], Callable, RV]]]


class Assignment:
    def __init__(
        self,
        functor: Callable,
        parents: Sequence[str],
        desc: Optional[str] = None,
    ):
        assert callable(functor), "Passed functor must be callable."
        self._descriptor = desc
        self.functor = functor
        self.parents_order = {
            par: pos for (par, pos) in zip(parents, range(len(parents)))
        }

    @property
    def descriptor(self):
        return self._descriptor

    def __len__(self):
        return len(self.parents_order)

    def __getitem__(self, parent: str):
        # returns the parameter position of the given parent
        return self.parents_order[parent]

    def __call__(self, *args, **kwargs):
        args = list(args) + [None] * (len(self) - len(args))
        for var in kwargs:
            if var in self.parents_order:
                args[self.parents_order[var]] = kwargs[var]
        return self.functor(*args)

    def __str__(self):
        if self.descriptor is None:
            return "__unknown__"
        return self.descriptor

    @staticmethod
    def noise_argname(
        name: Union[str, RV], node_name: Optional[Union[str, sympy.Symbol]] = None
    ):
        return (
            f"__"
            f"noise"
            f"{'/' if node_name is not None and bool(str(node_name)) else ''}{node_name}"
            f"{'/' if str(name) else ''}{name}"
            f"__"
        )


class SCM:
    """
    Class building a Structural Causal Model.

    With this class one can sample causally from the underlying graph and perform interventions. Do-interventions and
    soft-interventions, as well as more general interventions targeting one or more variables with arbitrary
    changes to the assignment and/or noise structure (including parent-child relationship changes) are supported.

    For visualization aid, the SCM can plot itself and also print a summary of itself to the console.

    Notes
    -----
    When constructing or intervening the user is responsible to guarantee acyclicity in the SCM. No checks are enabled
    at this stage.
    """

    # the attribute list that any given node in the graph has.
    (assignment_key, rv_key, noise_key, noise_repr_key) = (
        "assignment",
        "rv",
        "noise",
        "noise_repr",
    )

    class NodeView:
        """
        A view of the variable's associated attributes
        """

        def __init__(
            self,
            var: str,
            parents: List[str],
            assignment: Assignment,
            rv: RV,
            noise: RV,
            noise_repr: str,
        ):
            self.variable = var
            self.parents = parents
            self.assignment = assignment
            self.rv = rv
            self.noise = noise
            self.noise_repr = noise_repr

    def __init__(
        self,
        assignments: Union[AssignmentMap, AssignmentSeq],
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
            If not provided, defaults to the input names in the functional map.
        seed: (optional) str,
            Seeding the graph for reproducibility.
        scm_name: (optional) str,
            The name of the SCM. Default is 'Structural Causal Model'.
        """
        self.scm_name: str = scm_name
        self.rng_state = np.random.default_rng()  # we always have a local RNG machine
        self.seed(seed)

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

        # supply any variable name, that has not been assigned a different TeX name, with itself as TeX name.
        # This prevents missing labels in the plot method.
        if variable_tex_names is not None:
            for name in self.get_variables():
                if name not in variable_tex_names:
                    variable_tex_names[name] = name
            # the variable names as they can be used by the plot function to draw the names in TeX mode.
            self.var_draw_dict: Dict = variable_tex_names
        else:
            self.var_draw_dict = {name: name for name in self.get_variables()}

    def __getitem__(self, var):
        attr_dict = self.dag.nodes[var]
        return SCM.NodeView(var, self.dag.pred[var], **attr_dict)

    def __str__(self):
        return self.str()

    def __iter__(self):
        return self._causal_iterator()

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

            noise_gens = node_attr[self.noise_key]
            if noise_gens is not None:
                for noise_gen in noise_gens:
                    named_args[
                        Assignment.noise_argname(str(noise_gen), node_name=node)
                    ] = np.asarray(
                        list(sample(noise_gen, size=(n,), seed=seed)), dtype=float
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
        else:
            seed = np.random.default_rng(seed)
        vars_ordered = list(self._causal_iterator(variables))
        noise_iters: Dict[str, List[Tuple[Optional[str], Iterable]]] = dict()
        for node in vars_ordered:
            noise_gens = self.dag.nodes[node][self.noise_key]
            if noise_gens:
                noise_iters[node] = []
                for noise_gen in noise_gens:
                    noise_iter = sample_iter(
                        noise_gen,
                        seed=seed,
                    )

                    noise_iters[node].append(
                        (
                            Assignment.noise_argname(noise_gen, node_name=node),
                            noise_iter,
                        )
                    )
            else:
                noise_iters[node] = [(None, repeat(None))]

        if container is None:
            samples = {var: [] for var in vars_ordered}
        else:
            samples = container
        while True:
            for node in vars_ordered:
                node_attr = self.dag.nodes[node]

                named_args = dict()
                for pred in self.dag.predecessors(node):
                    named_args[pred] = samples[pred][-1]

                for noise_name, noise_gen in noise_iters[node]:
                    noise = next(noise_gen)
                    if noise is not None:
                        named_args[noise_name] = noise

                data = node_attr[self.assignment_key](**named_args)
                samples[node].append(data)
            yield samples

    def intervention(
        self,
        interventions: Dict[
            str,
            Tuple[Optional[List[str]], Optional[Union[str, Callable]], Optional[RV]],
        ],
    ):
        """
        Method to apply interventions on the specified variables.

        One can set any variable to a new functional and noise model, thus redefining its parents and their
        dependency structure. Using this method will enable the user to call sample, plot etc. on the SCM just like
        before, but with the altered SCM.
        In order to allow for undoing the intervention(s), the original state of the variables in the network is saved
        as backup and can be undone by calling ``undo_intervention``.

        Parameters
        ----------
        interventions: dict,
            the variables as keys and their new assignments as values. For the assignment structure
            one has to provide a tuple of length 3 with the following positions:
                1. The new parents of the assignment.
                   Pass 'None' to leave the original dependencies intact.
                2. The new assignment function. Pass as 'str' or 'callable'. If it is a callable, then either
                   the old parent structure or the new parent structure (if provided) has to be fit with the
                   positional input of the function.
                   Pass 'None' to leave the original assignment intact.
                3. The new noise distribution. Ascertain that the new distributions name symbol overlaps with
                   the previous name symbol.
                   Pass 'None' to leave the original noise distribution intact.
        """
        interventional_map = dict()

        for var, items in interventions.items():
            if var not in self.get_variables():
                logging.warning(f"Variable '{var}' not found in graph. Omitting it.")
                continue

            if not isinstance(items, Sequence):
                raise ValueError(
                    f"Intervention items container '{type(items)}' not supported."
                )

            items = [elem for elem in items]
            existing_attr = self[var]
            assert (
                len(items) == 3
            ), f"Items tuple of variable {var} has wrong length. Given {len(items)}, expected: 3"

            if items[2] is None:
                # no new noise distribution given
                items[2] = existing_attr.noise

            if items[1] is None:
                # no new assignment given
                items[1] = existing_attr.assignment

            # whether the assignment is new or old: we have to parse it
            if callable(items[1]) and items[0] is None:
                # Rebuild the parents for the assignment, because no new parent info is given,
                # but a callable assignment needs a parents list
                sorted_parents = sorted(
                    self.get_variable_args(var).items(),
                    key=lambda k: k[1],
                )
                items[0] = [
                    parent
                    for parent, _ in sorted_parents[
                        1:
                    ]  # element 0 is the noise name (must be removed)
                ]
            elif isinstance(items[1], str):
                # We reach this space only if a new assignment was provided, since existing assignments are already
                # converted to a callable.
                # In string assignments the parents list is deduced:
                items.pop(0)
            else:
                raise ValueError(
                    f"Variable {var} has been given an unsupported assignment type. "
                    f"Expected: str or callable, given: {type(items[1])}"
                )

            if var not in self.interventions_backup_attr:
                # the variable has NOT already been backed up. If we overwrite it now it is fine. If it had already been
                # in the backup, then we would need to pass on this step, in order to not overwrite the backup
                # (possibly with a previous intervention)
                self.interventions_backup_attr[var] = deepcopy(self.dag.nodes[var])

            if var not in self.interventions_backup_parent:
                # the same logic goes for the parents backup.
                parent_backup = list(self.dag.predecessors(var))
                self.interventions_backup_parent[var] = parent_backup
                self.dag.remove_edges_from([(parent, var) for parent in parent_backup])
            # patch up the attr dict as the provided items were merely strings and now need to be parsed by sympy.
            interventional_map[var] = items
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
            interventions_dict[var] = (None, str(val), None)
        self.intervention(interventions_dict)

    def soft_intervention(
        self,
        var_noise_pairs: Sequence[Tuple[str, RV]],
    ):
        """
        Perform noise interventions, i.e. modifying the noise variable of specific variables.

        Convenience wrapper around ``interventions`` method.

        Parameters
        ----------
        var_noise_pairs : Sequence of (variable, noise_model) tuples,
            the variables of which to modify the noise model.
        """
        interventions_dict = dict()
        for var, noise in var_noise_pairs:
            interventions_dict[var] = (None, None, noise)
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
        Check whether the current DAG is indeed a DAG
        Returns
        -------
        bool,
            A boolean indicating whether the current graphical model is indeed a DAG.
        """
        return nx.is_directed_acyclic_graph(self.dag)

    def insert(self, assignments: Union[AssignmentSeq, AssignmentMap]):
        """
        Method to insert variables into the graph. The passable assignments are the same as for the constructor of
        the SCM class.

        Parameters
        ----------
        assignments, Sequence of str or dictionary of nodes to assignment tuples
            The assignments to add to the graph.

        """
        if isinstance(assignments, Sequence):
            assignments = parse_assignments(assignments)
        elif not isinstance(assignments, Dict):
            raise ValueError(
                f"Assignments parameter accepts either a "
                f"Sequence[str] or "
                f"a {str(AssignmentMap).replace('typing.', '')}."
            )

        for node_name, assignment_pack in assignments.items():
            # a sequence of size 2 is expected to be (assignment string, noise model or List[noise model])
            if len(assignment_pack) == 2:
                assignment_str, noise_model_list = assignment_pack
                if isinstance(noise_model_list, RV):
                    noise_model_list = [noise_model_list]
                parents = extract_parents(assignment_str, noise_model_list)

                assignment_func = lambdify_assignment(
                    parents, assignment_str, noise_model_list
                )
                rv = sympify_assignment(self, parents, assignment_str, noise_model_list)

            # a sequence of size 3 is expected to be (parents list, assignment string, noise model)
            elif len(assignment_pack) == 3:
                parents, assignment_func, noise_model_list = assignment_pack
                if isinstance(noise_model_list, RV):
                    noise_model_list = [noise_model_list]
                assert callable(
                    assignment_func
                ), "Assignment tuple holds 3 elements, but the function entry is not callable."
                assignment_str = None
                try:
                    rv = sympify_assignment(
                        self, parents, assignment_func, noise_model_list
                    )
                except TypeError:
                    # RV construction only works with SymPy known functions.
                    # If foreign functions are provided (e.g. numpy) then the
                    # functionality of an actual RV of the assigned variable
                    # cannot be provided.
                    rv = []

            else:
                raise ValueError(
                    "Assignment entry must be a sequence of 2 or 3 entries."
                )

            if len(parents) > 0:
                for parent in parents:
                    self.dag.add_edge(parent, node_name)
            else:
                self.roots.append(node_name)

            # note that we have to ALWAYS ensure that the assignment func and the list of extended parents
            # (i.e. noise models + actual parent variables) is in exactly the same order as given to the
            # lambdifying call! Otherwise, the Assignment class will assign the wrong positional order to the
            # variables and call the functor incorrectly (since the order differs from when the sympy assignment
            # was lambdified)
            noise_and_parents = [
                Assignment.noise_argname(noise_model, node_name=node_name)
                for noise_model in noise_model_list
            ] + list(parents)
            assignment = Assignment(
                assignment_func, noise_and_parents, desc=assignment_str
            )
            attr_dict = {
                self.assignment_key: assignment,
                self.rv_key: rv,
                self.noise_key: noise_model_list,
                self.noise_repr_key: extract_rv_desc(noise_model_list),
            }

            self.dag.add_node(
                node_name,
                **attr_dict,
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
        tree structure often can't be computed. Therefore, this method relies on graphviz to compute a feasible
        representation of the causal graph.

        The graphviz package has been marked as an optional package for this module and therefore needs to be installed
        by the user.
        Note, that graphviz may require dependant libraries, e.g. libgraphviz-dev on linux.

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
            pos = hierarchy_pos(self.dag, root_node=self.roots[0])
        else:
            pos = graphviz_layout(self.dag, prog="dot")
        if draw_labels:
            labels = self.var_draw_dict
        else:
            labels = {}
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.set_title(self.scm_name)
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
        variables = self.get_variables()
        nr_vars = len(variables)
        lines = [
            f"Structural Causal Model of {nr_vars} "
            + ("variables: " if nr_vars > 1 else "variable: ")
            + ", ".join(variables[0:10])
            + (", ..." if nr_vars > 10 else ""),
            f"Variables with active interventions: {list(self.interventions_backup_attr.keys())}",
            "Assignments:",
        ]
        max_var_space = max([len(var_name) for var_name in variables])
        for node in self.dag.nodes:
            parents_vars = [pred for pred in self.dag.predecessors(node)]
            node_view = self[node]
            noise_vars = []
            for noise_symbol in node_view.noise:
                noise_str = (
                    str(noise_symbol)
                    if isinstance(noise_symbol, RV)
                    else Assignment.noise_argname("")
                )
                noise_vars.append(noise_str)
            parents_vars = noise_vars + parents_vars
            args_str = ", ".join(parents_vars)
            line = f"{str(node).rjust(max_var_space)} := f({args_str}) = {str(node_view.assignment)}"
            if noise_vars:
                # add explanation to the noise term
                noise_definitions = [
                    f"{noise_str} ~ {str(noise_repr)}"
                    for noise_str, noise_repr in zip(noise_vars, node_view.noise_repr)
                ]
                line += f"\t [ " f"{', '.join(noise_definitions)}" f" ]"
            lines.append(line)
        return "\n".join(lines)

    def get_variables(self, causal_order: bool = True) -> List[str]:
        """
        Get a list of the variables in the SCM.

        Parameters
        ----------
        causal_order: bool (optional),
            If True, the list is guaranteed to be in a causal dependency order starting with root nodes.
            Defaults to True.

        Returns
        -------
        list,
            the variables in the SCM.
        """
        if causal_order:
            return list(self._causal_iterator())
        else:
            return list(self.dag.nodes)

    def get_variable_args(self, variable: str):
        """
        Returns the input arguments of the node and their position as parameter in the assignment.

        Parameters
        ----------
        variable: str,
            the variable of interest.

        Returns
        -------
        dict,
            a dictionary with the arguments as keys and their position as values
        """
        return self[variable].assignment.parents_order

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

    def _causal_iterator(self, variables: Optional[Iterable] = None):
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
        if variables is None:
            for node in nx.topological_sort(self.dag):
                yield node
            return
        visited_nodes: Set = set()
        var_causal_priority: Dict = defaultdict(int)
        queue = deque([var for var in self.filter_variable_names(variables, self.dag)])
        while queue:
            nn = queue.popleft()
            # this line appears to be pointless, but is necessary to emplace the node 'nn' in the dict with its current
            # value, if already present, otherwise with the default value (0).
            var_causal_priority[nn] = var_causal_priority[nn]
            if nn not in visited_nodes:
                for parent in self.dag.predecessors(nn):
                    var_causal_priority[parent] = max(
                        var_causal_priority[parent], var_causal_priority[nn] + 1
                    )
                    queue.append(parent)
                visited_nodes.add(nn)
        for key, _ in sorted(var_causal_priority.items(), key=lambda x: -x[1]):
            yield key


def lambdify_assignment(
    parents: Iterable[str], assignment_str: str, noise_model_list: Optional[List[RV]]
):
    """
    Parse the provided assignment string with sympy and then lambdifies it, to be used as a normal function.

    Parameters
    ----------
    assignment_str: str,
        the assignment to parse.
    parents: Iterable,
        the parents' names.
    noise_model: RV,
        the random variable inside the assignment.

    Returns
    -------
    function,
        the lambdified assignment.
    """

    # the assignment string is sympified into an evaluable function only.
    # We therefore need all RandomSymbols as simple Symbols
    symbols = []
    lcls = locals()
    for noise_model in noise_model_list:
        symbols.append(noise_model)
        # Allocate the noise model variable as normal symbol. This is necessary for lambdifying.
        lcls[str(noise_model)] = noise_model
    for par in parents:
        lcls[par] = sympy.Symbol(par)
        symbols.append(lcls[par])
    assignment = sympy.sympify(eval(assignment_str))
    try:
        assignment = sympy.lambdify(symbols, assignment, "numpy")
    except (NameError, ValueError) as e:
        warnings.warn(
            f"The assignment string could not be resolved in numpy, the error message reads: {e}\n"
            f"Lambdifying without numpy.",
        )
        assignment = sympy.lambdify(symbols, assignment)

    return assignment


def sympify_assignment(
    scm: SCM,
    parents: Iterable[str],
    assignment: Union[str, Callable],
    noise_model_list: Optional[List[RV]],
):
    """
    Parse the provided assignment string with sympy and then sympify it, to use it as a random variable.

    Parameters
    ----------
    scm: SCM,
        The scm is needed for variable RV lookup.
    assignment: str,
        the assignment to parse.
    parents: Iterable,
        the parents' names.
    noise_model: RV,
        the random variable inside the assignment.

    Returns
    -------
    RV,
        the sympified assignment.
    """
    symbols = []
    # we allocate the symbols with the actual RVs to produce the assignment as RV itself
    for noise_model in noise_model_list:
        symbols.append(noise_model)
        # Allocate the noise model variable as random symbol
        exec(f"{str(noise_model)} = noise_model")
    for par in parents:
        par_rv = scm[par].rv
        exec(f"{str(par)} = par_rv")
        exec(f"symbols.append({str(par)})")
    if isinstance(assignment, str):
        rv = sympy.sympify(eval(assignment))
    elif callable(assignment):
        rv = assignment(*symbols)
    else:
        raise ValueError(
            f"Passed assignment has to be str or callable. Given: {type(assignment)}."
        )
    return rv


def extract_rv_desc(rvs: Optional[List[RV]]):
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
    all_descs = []
    if rvs is not None:
        for rv in rvs:

            def rv_string(rv):
                dist = rv.pspace.args[1]
                argnames = dist._argnames
                args = dist.args
                dist_name = str(dist).split("(", 1)[0].replace("Distribution", "")
                # sympy has its own data types 'Int' and 'Float' which are not python's 'int' and 'float'.
                # Those floats need help in order to be pretty printed so we defer to numpy's formatter, instead
                # in the case of floats. Ints are printed just fine
                kwargs_with_values = [
                    arg
                    + "="
                    + np.array2string(
                        np.asarray(float(val)),
                        formatter={"float_kind": lambda x: "%g" % x},
                    )
                    if isinstance(val, (float, sympy.Float))
                    else arg + "=" + str(val)
                    for arg, val in zip(argnames, args)
                ]
                return f"{dist_name}({', '.join(kwargs_with_values)})"

            desc = []
            if isinstance(rv, RV):
                desc.append(rv_string(rv))
            else:
                # append first its str definition
                desc.append(str(rv))
                desc.append("with")
                # now add all inner RVs
                arg_strs = []
                for random_symbol in random_symbols(rv):
                    arg_strs.append(f"{random_symbol} ~ {rv_string(random_symbol)}")

                desc.append(", ".join(arg_strs))
            all_descs.append(" ".join(desc))
        return all_descs
    else:
        return str(None)


def hierarchy_pos(
    graph: nx.Graph,
    root_node=None,
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

    if root_node is None:
        if isinstance(graph, nx.DiGraph):
            root_node = next(
                iter(nx.topological_sort(graph))
            )  # allows back compatibility with nx version 1.11
        else:
            root_node = np.random.choice(list(graph.nodes))

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

            leftmost_child = min(
                (x for x, y in [leaf_pos_[child] for child in children])
            )
            rightmost_child = max(
                (x for x, y in [leaf_pos_[child] for child in children])
            )
            leaf_pos_[root_] = ((leftmost_child + rightmost_child) / 2, vert_loc_)
        else:
            leaf_count = 1
            leaf_pos_[root_] = (leftmost_, vert_loc_)
        #        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
        print(leaf_count)
        return root_pos_, leaf_pos_, leaf_count

    x_center = width / 2.0
    if isinstance(graph, nx.DiGraph):
        leaf_count = len(
            [
                node
                for node in nx.descendants(graph, root_node)
                if graph.out_degree(node) == 0
            ]
        )
    elif isinstance(graph, nx.Graph):
        leaf_count = len(
            [
                node
                for node in nx.node_connected_component(graph, root_node)
                if graph.degree(node) == 1 and node != root_node
            ]
        )
    else:
        raise ValueError(
            "Passed graph is neither a networkx.DiGraph nor networkx.Graph."
        )
    root_pos, leaf_pos, leaf_count = __hierarchy_pos(
        graph,
        root_node,
        0,
        width,
        leaf_dx_=width * 1.0 / leaf_count,
        vert_gap_=vert_gap,
        vert_loc_=vert_loc,
        xcenter_=x_center,
    )
    pos = {}
    for node in root_pos:
        pos[node] = (
            leaf_vs_root_factor * leaf_pos[node][0]
            + (1 - leaf_vs_root_factor) * root_pos[node][0],
            leaf_pos[node][1],
        )
    x_max = max(x for x, y in pos.values())
    for node in pos:
        pos[node] = (pos[node][0] * width / x_max, pos[node][1])
    return pos
