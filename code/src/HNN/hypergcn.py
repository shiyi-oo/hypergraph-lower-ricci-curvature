"""Dependency-light HyperGCN encoder adapted from DHG-Bench."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .data import HypergraphData


def build_hypergcn_adjacency(
    graph: HypergraphData,
    *,
    mediators: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Build the normalized graph approximation used by HyperGCN.

    Each hyperedge selects the nodes with minimum and maximum projection onto
    one random vector.  With ``mediators=True``, the remaining nodes are linked
    to both extrema, following the HyperGCN mediator construction.
    """
    x = graph.x.detach().cpu()
    projection_vector = torch.rand(
        x.size(1), generator=generator, dtype=x.dtype
    )
    projection = x @ projection_vector
    node_index, edge_index = graph.hyperedge_index.cpu()
    edge_nodes: list[list[int]] = [[] for _ in range(graph.num_edges)]
    for node, edge in zip(node_index.tolist(), edge_index.tolist()):
        edge_nodes[edge].append(node)

    weights: defaultdict[tuple[int, int], float] = defaultdict(float)

    def add(source: int, target: int, weight: float) -> None:
        weights[(source, target)] += weight

    for nodes in edge_nodes:
        values = projection[nodes]
        minimum = nodes[int(values.argmin())]
        maximum = nodes[int(values.argmax())]

        if mediators:
            scale = 1.0 / max(2 * len(nodes) - 3, 1)
            add(maximum, minimum, scale)
            add(minimum, maximum, scale)
            for mediator in nodes:
                if mediator != minimum and mediator != maximum:
                    add(maximum, mediator, scale)
                    add(minimum, mediator, scale)
                    add(mediator, maximum, scale)
                    add(mediator, minimum, scale)
        else:
            scale = 1.0 / len(nodes)
            add(maximum, minimum, scale)
            add(minimum, maximum, scale)

    for node in range(graph.num_nodes):
        add(node, node, 1.0)

    pairs = list(weights)
    indices = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    values = torch.tensor([weights[pair] for pair in pairs])
    degree = torch.zeros(graph.num_nodes, dtype=values.dtype)
    degree.index_add_(0, indices[0], values)
    inverse_sqrt_degree = degree.clamp_min_(1e-12).rsqrt()
    values = (
        values
        * inverse_sqrt_degree[indices[0]]
        * inverse_sqrt_degree[indices[1]]
    )
    # The context also covers sparse tensors created internally by coalesce().
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(
            indices,
            values,
            (graph.num_nodes, graph.num_nodes),
        ).coalesce()


class HyperGCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        return torch.sparse.mm(adjacency, self.linear(x))


class HyperGCNEncoder(nn.Module):
    """Stacked HyperGCN layers using a precomputed graph approximation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.dropout = dropout
        widths = (
            [in_channels, out_channels]
            if num_layers == 1
            else [in_channels]
            + [hidden_channels] * (num_layers - 1)
            + [out_channels]
        )
        self.layers = nn.ModuleList(
            HyperGCNConv(left, right)
            for left, right in zip(widths[:-1], widths[1:])
        )

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self, graph: HypergraphData, x: Tensor | None = None
    ) -> Tensor:
        if graph.hypergcn_adjacency is None:
            raise ValueError(
                "HyperGCN requires a precomputed hypergcn_adjacency"
            )
        x = graph.x if x is None else x
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, graph.hypergcn_adjacency)
            x = F.relu(x)
            if layer_idx < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


__all__ = [
    "HyperGCNConv",
    "HyperGCNEncoder",
    "build_hypergcn_adjacency",
]
