from scm.functionals import *
from scm.noise import *
from scm import SCM

import operator
from functools import reduce
from typing import Optional, Collection, Type

import numpy as np


class RandomSCM(SCM):
    def __init__(
        self,
        nr_nodes: int,
        functional_functions: Optional[Collection[Type[Functional]]],
        noise_models: Optional[Collection[Noise]],
        dependency_layers: Optional[Collection[int]] = None,
        max_nr_parents_per_node: int = 10,
        seed: Optional[int] = None,
    ):
        rs = np.random.default_rng(seed)
        if dependency_layers is None:
            max_nr_layers = max(nr_nodes // rs.integers(1, 10), 2)
            max_nodes_per_layer = nr_nodes//max_nr_layers
            if max_nodes_per_layer == 1:
                dependency_layers = [1] * max_nr_layers
            else:
                dependency_layers = rs.integers(1, max_nodes_per_layer, size=max_nr_layers)
            dependency_layers[-1] += nr_nodes - sum(dependency_layers)

        assert (
            np.sum(dependency_layers) == nr_nodes
        ), "Sum of node layers and total number of nodes need to be equal."


        functional_map = dict()
        node_names = [f"X_{i}" for i in range(nr_nodes)]
        plot_names = {node_names[i]: "$X_{" + str(i) + "}$" for i in range(nr_nodes)}
        layers_cumsum = np.cumsum(dependency_layers)
        nodes_to_layer = {
            node: self._get_node_layer(node, layers_cumsum) for node in node_names
        }
        layers_to_nodes = dict()
        for i, layer in enumerate(layers_cumsum):
            if i == 0:
                layers_to_nodes[i] = node_names[0 : layers_cumsum[i]]
            else:
                layers_to_nodes[i] = node_names[layers_cumsum[i - 1] : layers_cumsum[i]]
        for node in node_names:
            node_layer = nodes_to_layer[node]
            potential_parents = set(
                reduce(
                    operator.add,
                    (layers_to_nodes[l] for l in range(node_layer)),
                    [],
                )
            ) - {node}
            if potential_parents:
                parents_probs = rs.random(len(potential_parents))
                parents_probs /= parents_probs.sum()
                parents = rs.choice(
                    list(potential_parents),
                    p=parents_probs,
                    replace=False,
                    size=rs.integers(
                        1, min(max_nr_parents_per_node, len(potential_parents))
                    ) if len(potential_parents) > 1 else 1,
                )
            else:
                parents = []

            functional_map.update(
                {
                    node: (
                        parents,
                        rs.choice(functional_functions, seed).random_factory(
                            len(parents)
                        ),
                        rs.choice(noise_models),
                    )
                }
            )
        super().__init__(functional_map=functional_map, variable_tex_names=plot_names)
        if seed is not None:
            self.reseed(seed)

    @staticmethod
    def _get_node_layer(node: str, layer_cumsum: Collection[int]):
        layer = 0
        node_numeric = int(node.split("_")[-1])
        for i, curr_sum in enumerate(layer_cumsum):
            if node_numeric < curr_sum:
                layer = i
                break
        return layer
