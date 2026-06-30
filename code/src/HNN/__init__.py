"""Unified HGNN and HyperGCN graph-embedding package."""

from .data import FeatureMode, HypergraphData, dataset_to_graphs, edges_to_data
from .hgnn import HGNNConv, HGNNEncoder
from .hypergcn import (
    HyperGCNConv,
    HyperGCNEncoder,
    build_hypergcn_adjacency,
)
from .trainer import (
    HNNConfig,
    HNNFitResult,
    build_encoder,
    fit_hgnn_embeddings,
    fit_hypergcn_embeddings,
    fit_hypergraph_embeddings,
    get_hypergraph_embeddings,
)

__all__ = [
    "FeatureMode",
    "HGNNConv",
    "HGNNEncoder",
    "HNNConfig",
    "HNNFitResult",
    "HyperGCNConv",
    "HyperGCNEncoder",
    "HypergraphData",
    "build_encoder",
    "build_hypergcn_adjacency",
    "dataset_to_graphs",
    "edges_to_data",
    "fit_hgnn_embeddings",
    "fit_hypergcn_embeddings",
    "fit_hypergraph_embeddings",
    "get_hypergraph_embeddings",
]
