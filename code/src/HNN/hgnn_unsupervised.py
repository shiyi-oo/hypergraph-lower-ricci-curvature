"""Backward-compatible high-level imports for the unified HNN trainer."""

from typing import Any, Sequence

from .data import HypergraphData, dataset_to_graphs, edges_to_data
from .trainer import (
    HNNConfig,
    HNNFitResult,
    fit_hgnn_embeddings,
    get_hypergraph_embeddings,
)

HGNNConfig = HNNConfig
HGNNFitResult = HNNFitResult
HGNNGraphData = HypergraphData


def edges_to_hgnn_data(
    edges: Sequence[Sequence[Any]],
    *,
    hypergraph_id: Any = None,
    feature_mode: str = "degree",
) -> HypergraphData:
    nodes = list(dict.fromkeys(node for edge in edges for node in edge))
    vocabulary = {node: idx for idx, node in enumerate(nodes)}
    return edges_to_data(
        edges,
        vocabulary=vocabulary,
        hypergraph_id=hypergraph_id,
        feature_mode=feature_mode,
    )


def dataset_to_hgnn_graphs(
    dataset: Any,
    *,
    feature_mode: str = "degree",
) -> list[HypergraphData]:
    return dataset_to_graphs(dataset, feature_mode=feature_mode)

__all__ = [
    "HGNNConfig",
    "HGNNFitResult",
    "HGNNGraphData",
    "dataset_to_hgnn_graphs",
    "edges_to_hgnn_data",
    "fit_hgnn_embeddings",
    "get_hypergraph_embeddings",
]
