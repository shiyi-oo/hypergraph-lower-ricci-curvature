"""HGNN encoder implemented with native PyTorch reductions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .data import HypergraphData


def _sum_by_index(values: Tensor, index: Tensor, size: int) -> Tensor:
    output = values.new_zeros((size, values.size(-1)))
    output.index_add_(0, index, values)
    return output


class HGNNConv(nn.Module):
    """The propagation ``D^-1/2 H B^-1 H^T D^-1/2 X Theta``."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        node_index, edge_index = hyperedge_index
        num_nodes = x.size(0)
        num_edges = int(edge_index.max()) + 1

        node_degree = torch.bincount(
            node_index, minlength=num_nodes
        ).to(x.dtype).clamp_min_(1)
        edge_degree = torch.bincount(
            edge_index, minlength=num_edges
        ).to(x.dtype).clamp_min_(1)

        x = self.linear(x)
        x = x * node_degree.rsqrt()[:, None]
        edge_embeddings = _sum_by_index(
            x[node_index], edge_index, num_edges
        )
        edge_embeddings = edge_embeddings / edge_degree[:, None]
        output = _sum_by_index(
            edge_embeddings[edge_index], node_index, num_nodes
        )
        return output * node_degree.rsqrt()[:, None]


class HGNNEncoder(nn.Module):
    """Stacked HGNN convolutions with a graph-encoder interface."""

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
            HGNNConv(left, right)
            for left, right in zip(widths[:-1], widths[1:])
        )

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self, graph: HypergraphData, x: Tensor | None = None
    ) -> Tensor:
        x = graph.x if x is None else x
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, graph.hyperedge_index)
            if layer_idx < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


__all__ = ["HGNNConv", "HGNNEncoder"]
