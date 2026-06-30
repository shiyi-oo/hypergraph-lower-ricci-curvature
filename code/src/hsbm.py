"""Generators for uniform and non-uniform HSBM toy hypergraphs.

The uniform model independently samples every possible edge of one fixed size.
The non-uniform model preserves the sequential growth process used by the
original project: each edge starts from one random node and grows by testing
the remaining nodes one at a time.

Both generators are intended for small synthetic examples.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from numbers import Integral, Real
import random

import numpy as np

HSBMResult = tuple[list[list[int]], list[int], list[int]]

SAMPLING_STRATEGIES = {"first", "weighted", "frequent", "max", "min"}
ORDER_STRATEGIES = {"random", "community-order", "fixed"}

# The old Generator sampled one name per node from a 293-item name pool before
# generating any edges.  Replaying those draws preserves its seeded datasets
# without retaining names or the legacy NetworkX representation.
_LEGACY_NAME_POOL_SIZE = 293

__all__ = [
    "HSBMResult",
    "generate_nonuniform_hsbm",
    "generate_uniform_hsbm",
]


def generate_uniform_hsbm(
    k_uniform: int,
    block_sizes: Sequence[int],
    community_edge_probs: Sequence[float],
    seed: int | None = 123,
) -> HSBMResult:
    """Generate a ``k_uniform``-uniform Bernoulli HSBM.

    Every possible edge of size ``k_uniform`` is sampled independently.  If
    an edge contains nodes from ``r`` distinct communities, its inclusion
    probability is ``community_edge_probs[r - 1]``.

    Examples
    --------
    >>> edges, labels, edge_community_counts = generate_uniform_hsbm(
    ...     k_uniform=3,
    ...     block_sizes=[5, 5],
    ...     community_edge_probs=[0.4, 0.02],
    ...     seed=7,
    ... )
    >>> all(len(edge) == 3 for edge in edges)
    True
    """
    blocks = _validate_block_sizes(block_sizes)
    number_of_nodes = sum(blocks)
    edge_size = _validate_uniform_edge_size(k_uniform, number_of_nodes)
    probabilities = _validate_community_probabilities(
        community_edge_probs,
        required_count=min(edge_size, len(blocks)),
    )

    node_labels = _node_labels(blocks)
    rng = np.random.default_rng(seed)
    hyperedges: list[list[int]] = []
    edge_community_counts: list[int] = []

    for edge in combinations(range(number_of_nodes), edge_size):
        number_of_edge_communities = len({node_labels[node] for node in edge})
        if rng.random() < probabilities[number_of_edge_communities - 1]:
            hyperedges.append(list(edge))
            edge_community_counts.append(number_of_edge_communities)

    return hyperedges, node_labels, edge_community_counts


def generate_nonuniform_hsbm(
    n_hyperedges: int,
    block_sizes: Sequence[int],
    p_edge_intra: float,
    p_edge_inter: float,
    sampling_strategy: str = "weighted",
    order_strategy: str = "random",
    seed: int | None = 123,
) -> HSBMResult:
    """Generate a non-uniform HSBM using sequential edge growth.

    Each of the ``n_hyperedges`` edges begins with one uniformly sampled node.
    The other nodes are visited sequentially and conditionally added according
    to ``sampling_strategy``.  Consequently, edge sizes are random and each
    generated edge contains at least one node.  Separate generated edges may
    have identical memberships, matching the behavior of the original model.

    Sampling strategies
    -------------------
    ``first``:
        Compare the candidate's community with the first node's community.
    ``weighted``:
        Average the intra/inter probabilities between the candidate and all
        nodes already in the edge.
    ``frequent``:
        Compare with the most frequent community currently in the edge.
    ``max``:
        Use the intra probability if any current node shares the candidate's
        community.
    ``min``:
        Use the inter probability if any current node has another community.

    The order strategy is ``random`` for a new permutation per edge,
    ``community-order`` for node-index order, or ``fixed`` for one random
    permutation shared by all generated edges.

    Examples
    --------
    >>> edges, labels, edge_community_counts = generate_nonuniform_hsbm(
    ...     n_hyperedges=20,
    ...     block_sizes=[10, 10],
    ...     p_edge_intra=0.2,
    ...     p_edge_inter=0.02,
    ...     sampling_strategy="frequent",
    ...     seed=7,
    ... )
    >>> len(edges)
    20
    >>> all(len(edge) >= 1 for edge in edges)
    True
    """
    blocks = _validate_block_sizes(block_sizes)
    edge_count = _validate_hyperedge_count(n_hyperedges)
    p_intra = _validate_probability(p_edge_intra, "p_edge_intra")
    p_inter = _validate_probability(p_edge_inter, "p_edge_inter")
    _validate_strategy(sampling_strategy, SAMPLING_STRATEGIES, "sampling_strategy")
    _validate_strategy(order_strategy, ORDER_STRATEGIES, "order_strategy")

    node_labels = _node_labels(blocks)
    number_of_nodes = len(node_labels)
    nodes = list(range(number_of_nodes))
    rng = random.Random(seed)

    # Preserve the old Generator's random-state progression.  It created a
    # fixed order even when another order strategy was used, then generated
    # one decorative animal name per node.
    fixed_order = nodes.copy()
    rng.shuffle(fixed_order)
    for _ in nodes:
        rng.randrange(_LEGACY_NAME_POOL_SIZE)

    hyperedges: list[list[int]] = []
    edge_community_counts: list[int] = []

    for _ in range(edge_count):
        node_order = _node_order(nodes, order_strategy, fixed_order, rng)
        first_node = rng.choice(node_order)
        edge = [first_node]

        for node in node_order:
            if node == first_node:
                continue

            inclusion_probability = _node_inclusion_probability(
                node=node,
                edge=edge,
                node_labels=node_labels,
                p_intra=p_intra,
                p_inter=p_inter,
                sampling_strategy=sampling_strategy,
            )
            if rng.random() < inclusion_probability:
                edge.append(node)

        hyperedges.append(edge)
        edge_community_counts.append(len({node_labels[node] for node in edge}))

    return hyperedges, node_labels, edge_community_counts


def _node_inclusion_probability(
    node: int,
    edge: Sequence[int],
    node_labels: Sequence[int],
    p_intra: float,
    p_inter: float,
    sampling_strategy: str,
) -> float:
    candidate_community = node_labels[node]
    edge_communities = [node_labels[member] for member in edge]

    if sampling_strategy == "first":
        return p_intra if candidate_community == edge_communities[0] else p_inter

    if sampling_strategy == "weighted":
        same_community_count = edge_communities.count(candidate_community)
        same_community_fraction = same_community_count / len(edge)
        return (
            same_community_fraction * p_intra
            + (1.0 - same_community_fraction) * p_inter
        )

    if sampling_strategy == "frequent":
        most_frequent_community = max(
            set(edge_communities),
            key=edge_communities.count,
        )
        return p_intra if candidate_community == most_frequent_community else p_inter

    if sampling_strategy == "max":
        return p_intra if candidate_community in edge_communities else p_inter

    # ``min``: one different community is enough to use the inter probability.
    return (
        p_inter
        if any(community != candidate_community for community in edge_communities)
        else p_intra
    )


def _node_order(
    nodes: list[int],
    order_strategy: str,
    fixed_order: list[int],
    rng: random.Random,
) -> list[int]:
    if order_strategy == "random":
        node_order = nodes.copy()
        rng.shuffle(node_order)
        return node_order
    if order_strategy == "fixed":
        return fixed_order
    return nodes


def _node_labels(block_sizes: Sequence[int]) -> list[int]:
    return [
        community
        for community, block_size in enumerate(block_sizes)
        for _ in range(block_size)
    ]


def _validate_block_sizes(block_sizes: Sequence[int]) -> tuple[int, ...]:
    blocks = tuple(block_sizes)
    if not blocks:
        raise ValueError("block_sizes must contain at least one community")
    if any(
        isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
        for size in blocks
    ):
        raise ValueError("every block size must be a positive integer")
    return tuple(int(size) for size in blocks)


def _validate_uniform_edge_size(k_uniform: int, number_of_nodes: int) -> int:
    if (
        isinstance(k_uniform, bool)
        or not isinstance(k_uniform, Integral)
        or not 2 <= k_uniform <= number_of_nodes
    ):
        raise ValueError(
            "k_uniform must be an integer between 2 and the number of nodes"
        )
    return int(k_uniform)


def _validate_hyperedge_count(n_hyperedges: int) -> int:
    if (
        isinstance(n_hyperedges, bool)
        or not isinstance(n_hyperedges, Integral)
        or n_hyperedges < 0
    ):
        raise ValueError("n_hyperedges must be a non-negative integer")
    return int(n_hyperedges)


def _validate_community_probabilities(
    probabilities: Sequence[float],
    required_count: int,
) -> tuple[float, ...]:
    probs = tuple(probabilities)
    if len(probs) < required_count:
        raise ValueError(
            f"at least {required_count} community probabilities are required; "
            f"received {len(probs)}"
        )
    return tuple(
        _validate_probability(probability, "community_edge_probs")
        for probability in probs
    )


def _validate_probability(probability: float, parameter_name: str) -> float:
    if (
        isinstance(probability, bool)
        or not isinstance(probability, Real)
        or not 0 <= probability <= 1
    ):
        raise ValueError(f"{parameter_name} must be a number between 0 and 1")
    return float(probability)


def _validate_strategy(
    strategy: str,
    valid_strategies: set[str],
    parameter_name: str,
) -> None:
    if strategy not in valid_strategies:
        choices = ", ".join(sorted(valid_strategies))
        raise ValueError(f"{parameter_name} must be one of: {choices}")
