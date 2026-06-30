"""Data conversion shared by the HGNN and HyperGCN encoders."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Sequence

import torch
from torch import Tensor


FeatureMode = Literal["auto", "constant", "degree", "identity", "identity_degree"]


@dataclass(frozen=True)
class HypergraphData:
    """Tensor representation of one independent hypergraph."""

    x: Tensor
    hyperedge_index: Tensor
    num_nodes: int
    num_edges: int
    hypergraph_id: Any = None
    node_ids: tuple[Any, ...] = ()
    node_curvature: Tensor | None = None
    hypergcn_adjacency: Tensor | None = None

    def to(self, device: torch.device | str) -> "HypergraphData":
        adjacency = self.hypergcn_adjacency
        return replace(
            self,
            x=self.x.to(device),
            hyperedge_index=self.hyperedge_index.to(device),
            node_curvature=(
                None
                if self.node_curvature is None
                else self.node_curvature.to(device)
            ),
            hypergcn_adjacency=None if adjacency is None else adjacency.to(device),
        )

    def with_hypergcn_adjacency(self, adjacency: Tensor) -> "HypergraphData":
        return replace(self, hypergcn_adjacency=adjacency)


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _clean_edges(edges: Sequence[Sequence[Any]]) -> list[list[Any]]:
    if not edges:
        raise ValueError("a hypergraph must contain at least one hyperedge")

    clean: list[list[Any]] = []
    for edge_idx, edge in enumerate(edges):
        members = _ordered_unique(edge)
        if not members:
            raise ValueError(f"hyperedge {edge_idx} is empty")
        clean.append(members)
    return clean


def _finalize_node_curvature(
    graph: HypergraphData,
    mean: Tensor,
    scale: Tensor,
    standardize: bool,
) -> HypergraphData:
    if graph.node_curvature is None:
        return graph
    imputed = torch.where(
        torch.isfinite(graph.node_curvature),
        graph.node_curvature,
        mean,
    )
    feature = (imputed - mean) / scale if standardize else imputed
    return replace(
        graph,
        node_curvature=imputed,
        x=torch.cat((graph.x[:, :-1], feature[:, None]), dim=1),
    )


def _node_features(
    nodes: Sequence[Any],
    hyperedge_index: Tensor,
    *,
    feature_mode: FeatureMode,
    vocabulary: dict[Any, int],
) -> Tensor:
    degree = torch.bincount(
        hyperedge_index[0], minlength=len(nodes)
    ).to(torch.float32)
    log_degree = torch.log1p(degree)
    log_degree = (log_degree - log_degree.mean()) / (
        log_degree.std(unbiased=False) + 1e-8
    )

    if feature_mode == "constant":
        return torch.ones((len(nodes), 1), dtype=torch.float32)
    if feature_mode == "degree":
        return torch.stack((torch.ones_like(degree), log_degree), dim=1)

    identity = torch.zeros(
        (len(nodes), len(vocabulary)), dtype=torch.float32
    )
    identity[
        torch.arange(len(nodes)),
        torch.tensor([vocabulary[node] for node in nodes]),
    ] = 1.0
    if feature_mode == "identity":
        return identity
    if feature_mode == "identity_degree":
        return torch.cat((identity, log_degree[:, None]), dim=1)
    raise ValueError(f"unsupported feature_mode: {feature_mode}")


def edges_to_data(
    edges: Sequence[Sequence[Any]],
    *,
    vocabulary: dict[Any, int],
    hypergraph_id: Any = None,
    feature_mode: FeatureMode = "degree",
    hyperedge_curvatures: Sequence[float] | Tensor | None = None,
) -> HypergraphData:
    """Convert one hyperedge list to the common encoder input format.

    When edge curvatures are supplied, the finite incident-edge mean

    ``c_v = mean(c_e for e containing v)``

    is appended as one additional node feature. Dataset-level conversion
    imputes nodes without any finite incident curvature using the global mean.
    """
    if feature_mode == "auto":
        raise ValueError("feature_mode='auto' must be resolved at dataset level")

    clean_edges = _clean_edges(edges)
    nodes = _ordered_unique(node for edge in clean_edges for node in edge)
    local_node = {node: idx for idx, node in enumerate(nodes)}
    incidence = [
        (local_node[node], edge_idx)
        for edge_idx, edge in enumerate(clean_edges)
        for node in edge
    ]
    hyperedge_index = torch.tensor(
        incidence, dtype=torch.long
    ).t().contiguous()
    x = _node_features(
        nodes,
        hyperedge_index,
        feature_mode=feature_mode,
        vocabulary=vocabulary,
    )
    node_curvature = None
    if hyperedge_curvatures is not None:
        edge_curvature = torch.as_tensor(
            hyperedge_curvatures, dtype=torch.float32
        ).flatten()
        if edge_curvature.numel() != len(clean_edges):
            raise ValueError(
                "hyperedge_curvatures must have one value per hyperedge"
            )
        if torch.isinf(edge_curvature).any():
            raise ValueError("hyperedge_curvatures contains infinite values")
        node_index, edge_index = hyperedge_index
        valid_edge = torch.isfinite(edge_curvature)
        node_curvature = torch.zeros(len(nodes), dtype=torch.float32)
        node_curvature.index_add_(
            0,
            node_index,
            torch.nan_to_num(edge_curvature[edge_index], nan=0.0),
        )
        valid_count = torch.zeros(len(nodes), dtype=torch.float32)
        valid_count.index_add_(
            0, node_index, valid_edge[edge_index].to(torch.float32)
        )
        node_curvature = node_curvature / valid_count.clamp_min_(1)
        node_curvature[valid_count == 0] = float("nan")
        x = torch.cat((x, node_curvature[:, None]), dim=1)

    return HypergraphData(
        x=x,
        hyperedge_index=hyperedge_index,
        num_nodes=len(nodes),
        num_edges=len(clean_edges),
        hypergraph_id=hypergraph_id,
        node_ids=tuple(nodes),
        node_curvature=node_curvature,
    )


def dataset_to_graphs(
    dataset: Any,
    *,
    feature_mode: FeatureMode = "auto",
    max_identity_features: int = 512,
    hyperedge_curvatures: Sequence[float] | Tensor | None = None,
    standardize_node_curvature: bool = True,
) -> list[HypergraphData]:
    """Split a ``HypergraphDataset`` into independent graph objects.

    ``auto`` uses identity-plus-degree features when the global node vocabulary
    has at most ``max_identity_features`` entries, and degree features
    otherwise.  This gives MUS its 61 shared node identities without making
    large-vocabulary datasets allocate impractically wide one-hot matrices.
    """
    hyperedges = dataset.hyperedges
    graph_ids = dataset.hypergraph_idx
    edge_curvature = None
    if hyperedge_curvatures is not None:
        edge_curvature = torch.as_tensor(
            hyperedge_curvatures, dtype=torch.float32
        ).flatten()
        if edge_curvature.numel() != len(hyperedges):
            raise ValueError(
                "hyperedge_curvatures must align with dataset.hyperedges"
            )
        if torch.isinf(edge_curvature).any():
            raise ValueError("hyperedge_curvatures contains infinite values")
        if not torch.isfinite(edge_curvature).any():
            raise ValueError("hyperedge_curvatures has no finite values")
    vocabulary_nodes = _ordered_unique(
        node for edge in hyperedges for node in edge
    )
    vocabulary = {
        node: idx for idx, node in enumerate(vocabulary_nodes)
    }

    resolved_mode: FeatureMode = feature_mode
    if feature_mode == "auto":
        resolved_mode = (
            "identity_degree"
            if len(vocabulary) <= max_identity_features
            else "degree"
        )

    if graph_ids is None:
        groups = [(0, list(range(len(hyperedges))))]
    else:
        if len(graph_ids) != len(hyperedges):
            raise ValueError(
                "hypergraph_idx must have the same length as hyperedges"
            )
        grouped: OrderedDict[Any, list[int]] = OrderedDict()
        for edge_idx, graph_id in enumerate(graph_ids):
            grouped.setdefault(graph_id, []).append(edge_idx)
        groups = list(grouped.items())

    graphs = [
        edges_to_data(
            [hyperedges[edge_idx] for edge_idx in edge_indices],
            vocabulary=vocabulary,
            hypergraph_id=graph_id,
            feature_mode=resolved_mode,
            hyperedge_curvatures=(
                None
                if edge_curvature is None
                else edge_curvature[edge_indices]
            ),
        )
        for graph_id, edge_indices in groups
    ]

    if edge_curvature is not None:
        all_node_curvatures = torch.cat([
            graph.node_curvature for graph in graphs
            if graph.node_curvature is not None
        ])
        finite_node_curvatures = all_node_curvatures[
            torch.isfinite(all_node_curvatures)
        ]
        mean = finite_node_curvatures.mean()
        scale = finite_node_curvatures.std(unbiased=False) + 1e-8
        graphs = [
            _finalize_node_curvature(
                graph, mean, scale, standardize_node_curvature
            )
            for graph in graphs
        ]
    return graphs


__all__ = [
    "FeatureMode",
    "HypergraphData",
    "dataset_to_graphs",
    "edges_to_data",
]
