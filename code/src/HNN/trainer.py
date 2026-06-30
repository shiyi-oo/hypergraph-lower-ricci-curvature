"""Unified unsupervised training entry point for HGNN and HyperGCN."""

from __future__ import annotations

import random
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .data import FeatureMode, HypergraphData, dataset_to_graphs
from .hgnn import HGNNEncoder
from .hypergcn import HyperGCNEncoder, build_hypergcn_adjacency


Method = Literal["hgnn", "hypergcn"]
PoolingMode = Literal["mean", "max"]


@dataclass(frozen=True)
class HNNConfig:
    """Model and graph-contrastive training settings."""

    embedding_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    pooling: PoolingMode = "mean"
    projection_dim: int = 64
    projection_hidden_dim: int = 128
    feature_mask_rate: float = 0.15
    temperature: float = 0.2
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float | None = 5.0
    hypergcn_mediators: bool = True
    normalize_embeddings: bool = True


@dataclass(frozen=True)
class HNNFitResult:
    """Trained encoder and one embedding per input hypergraph."""

    method: Method
    model: nn.Module
    projector: nn.Module
    embeddings: np.ndarray
    hypergraph_ids: tuple[Any, ...]
    history: tuple[float, ...]
    config: HNNConfig


class ProjectionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


def _normalize_method(method: str) -> Method:
    normalized = method.lower().replace("-", "").replace("_", "")
    if normalized == "hgnn":
        return "hgnn"
    if normalized == "hypergcn":
        return "hypergcn"
    raise ValueError("method must be 'hgnn' or 'hypergcn'")


def _validate_config(config: HNNConfig) -> None:
    if min(
        config.embedding_dim,
        config.hidden_dim,
        config.num_layers,
        config.projection_dim,
        config.projection_hidden_dim,
        config.batch_size,
        config.epochs,
    ) < 1:
        raise ValueError("dimensions, layers, batch_size, and epochs must be positive")
    if config.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for contrastive training")
    if not 0 <= config.feature_mask_rate < 1:
        raise ValueError("feature_mask_rate must be in [0, 1)")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive")
    if config.pooling not in {"mean", "max"}:
        raise ValueError("pooling must be 'mean' or 'max'")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
    else:
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resolved.type == "mps":
        raise ValueError(
            "HyperGCN sparse operations are not reliably supported on MPS; "
            "use device='cpu'"
        )
    return resolved


def build_encoder(
    method: str,
    num_features: int,
    config: HNNConfig,
) -> nn.Module:
    """Build either encoder with the same architecture arguments."""
    resolved_method = _normalize_method(method)
    encoder_type = HGNNEncoder if resolved_method == "hgnn" else HyperGCNEncoder
    return encoder_type(
        in_channels=num_features,
        hidden_channels=config.hidden_dim,
        out_channels=config.embedding_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def _prepare_graphs(
    graphs: Sequence[HypergraphData],
    method: Method,
    config: HNNConfig,
    random_state: int,
    verbose: bool,
) -> list[HypergraphData]:
    if method == "hgnn":
        return list(graphs)

    prepared: list[HypergraphData] = []
    for graph_idx, graph in enumerate(graphs):
        generator = torch.Generator().manual_seed(random_state + graph_idx)
        adjacency = build_hypergcn_adjacency(
            graph,
            mediators=config.hypergcn_mediators,
            generator=generator,
        )
        prepared.append(graph.with_hypergcn_adjacency(adjacency))
        if verbose and (
            graph_idx == 0
            or (graph_idx + 1) % 250 == 0
            or graph_idx + 1 == len(graphs)
        ):
            print(
                f"HyperGCN preprocessing {graph_idx + 1}/{len(graphs)}"
            )
    return prepared


def _pool(node_embeddings: Tensor, mode: PoolingMode) -> Tensor:
    if mode == "mean":
        return node_embeddings.mean(dim=0)
    if mode == "max":
        return node_embeddings.max(dim=0).values
    raise ValueError("pooling must be 'mean' or 'max'")


def _mask_features(x: Tensor, rate: float) -> Tensor:
    if rate == 0:
        return x
    keep = (torch.rand_like(x) >= rate).to(x.dtype)
    return x * keep / (1.0 - rate)


def contrastive_loss(z1: Tensor, z2: Tensor, temperature: float) -> Tensor:
    """Symmetric cross-view InfoNCE loss for graph embeddings."""
    if z1.shape != z2.shape or z1.dim() != 2:
        raise ValueError("z1 and z2 must have the same [batch, feature] shape")
    if z1.size(0) < 2:
        raise ValueError("contrastive loss requires at least two graphs")
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.t() / temperature
    targets = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (
        F.cross_entropy(logits, targets)
        + F.cross_entropy(logits.t(), targets)
    )


def _batches(order: list[int], batch_size: int) -> list[list[int]]:
    batches = [
        order[start : start + batch_size]
        for start in range(0, len(order), batch_size)
    ]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def train_unsupervised(
    model: nn.Module,
    projector: nn.Module,
    graphs: Sequence[HypergraphData],
    *,
    config: HNNConfig,
    device: torch.device | str | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> list[float]:
    """Train a shared graph encoder using two feature-masked views."""
    if len(graphs) < 2:
        raise ValueError("contrastive training requires at least two hypergraphs")
    _validate_config(config)
    _set_seed(random_state)
    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    projector.to(resolved_device)
    optimizer = torch.optim.Adam(
        [*model.parameters(), *projector.parameters()],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[float] = []

    for epoch in range(config.epochs):
        model.train()
        projector.train()
        order = torch.randperm(len(graphs)).tolist()
        total_loss = 0.0
        batches = _batches(order, config.batch_size)

        for batch_indices in batches:
            optimizer.zero_grad(set_to_none=True)
            first_view: list[Tensor] = []
            second_view: list[Tensor] = []
            for graph_idx in batch_indices:
                graph = graphs[graph_idx].to(resolved_device)
                node_z1 = model(
                    graph,
                    x=_mask_features(graph.x, config.feature_mask_rate),
                )
                node_z2 = model(
                    graph,
                    x=_mask_features(graph.x, config.feature_mask_rate),
                )
                first_view.append(_pool(node_z1, config.pooling))
                second_view.append(_pool(node_z2, config.pooling))

            z1 = projector(torch.stack(first_view))
            z2 = projector(torch.stack(second_view))
            loss = contrastive_loss(z1, z2, config.temperature)
            loss.backward()
            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(
                    [*model.parameters(), *projector.parameters()],
                    config.gradient_clip,
                )
            optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / len(batches)
        history.append(mean_loss)
        if verbose and (
            epoch == 0
            or (epoch + 1) % 5 == 0
            or epoch + 1 == config.epochs
        ):
            print(
                f"Epoch {epoch + 1:03d}/{config.epochs}: "
                f"loss={mean_loss:.6f}"
            )
    return history


@torch.no_grad()
def get_hypergraph_embeddings(
    model: nn.Module,
    graphs: Sequence[HypergraphData],
    *,
    pooling: PoolingMode = "mean",
    normalize: bool = True,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Pool one fixed-width embedding for every hypergraph."""
    if not graphs:
        raise ValueError("graphs must not be empty")
    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    was_training = model.training
    model.eval()
    embeddings: list[Tensor] = []
    for graph in graphs:
        graph = graph.to(resolved_device)
        embedding = _pool(model(graph), pooling)
        if normalize:
            embedding = F.normalize(embedding, dim=0)
        embeddings.append(embedding.cpu())
    if was_training:
        model.train()
    return torch.stack(embeddings).numpy()


def fit_hypergraph_embeddings(
    dataset: Any,
    *,
    method: str,
    config: HNNConfig | None = None,
    feature_mode: FeatureMode = "auto",
    max_identity_features: int = 512,
    hyperedge_curvatures: Sequence[float] | Tensor | None = None,
    standardize_node_curvature: bool = True,
    device: torch.device | str | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> HNNFitResult:
    """Unified entry point for unsupervised HGNN and HyperGCN embeddings."""
    config = HNNConfig() if config is None else config
    _validate_config(config)
    resolved_method = _normalize_method(method)
    _set_seed(random_state)

    graphs = dataset_to_graphs(
        dataset,
        feature_mode=feature_mode,
        max_identity_features=max_identity_features,
        hyperedge_curvatures=hyperedge_curvatures,
        standardize_node_curvature=standardize_node_curvature,
    )
    if len(graphs) < 2:
        raise ValueError("dataset must contain at least two hypergraphs")
    num_features = graphs[0].x.size(1)
    if any(graph.x.size(1) != num_features for graph in graphs):
        raise ValueError("all hypergraphs must have the same feature width")

    graphs = _prepare_graphs(
        graphs,
        resolved_method,
        config,
        random_state,
        verbose,
    )
    model = build_encoder(resolved_method, num_features, config)
    projector = ProjectionHead(
        config.embedding_dim,
        config.projection_hidden_dim,
        config.projection_dim,
    )
    history = train_unsupervised(
        model,
        projector,
        graphs,
        config=config,
        device=device,
        random_state=random_state,
        verbose=verbose,
    )
    embeddings = get_hypergraph_embeddings(
        model,
        graphs,
        pooling=config.pooling,
        normalize=config.normalize_embeddings,
        device=device,
    )
    return HNNFitResult(
        method=resolved_method,
        model=model,
        projector=projector,
        embeddings=embeddings,
        hypergraph_ids=tuple(graph.hypergraph_id for graph in graphs),
        history=tuple(history),
        config=config,
    )


def _fit_method(
    dataset: Any,
    method: Method,
    config: HNNConfig | None,
    kwargs: dict[str, Any],
) -> HNNFitResult:
    """Accept config fields either in ``HNNConfig`` or as convenience kwargs."""
    config_names = {field.name for field in fields(HNNConfig)}
    config_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in config_names
    }
    if config_kwargs.get("pooling") == "node_mean":
        config_kwargs["pooling"] = "mean"
    if config_kwargs.get("pooling") == "node_max":
        config_kwargs["pooling"] = "max"
    resolved_config = HNNConfig() if config is None else config
    if config_kwargs:
        resolved_config = replace(resolved_config, **config_kwargs)
    return fit_hypergraph_embeddings(
        dataset,
        method=method,
        config=resolved_config,
        **kwargs,
    )


def fit_hgnn_embeddings(
    dataset: Any,
    *,
    config: HNNConfig | None = None,
    **kwargs: Any,
) -> HNNFitResult:
    """Convenience wrapper for ``method='hgnn'``."""
    return _fit_method(dataset, "hgnn", config, dict(kwargs))


def fit_hypergcn_embeddings(
    dataset: Any,
    *,
    config: HNNConfig | None = None,
    **kwargs: Any,
) -> HNNFitResult:
    """Convenience wrapper for ``method='hypergcn'``."""
    return _fit_method(dataset, "hypergcn", config, dict(kwargs))


__all__ = [
    "HNNConfig",
    "HNNFitResult",
    "Method",
    "PoolingMode",
    "ProjectionHead",
    "build_encoder",
    "contrastive_loss",
    "fit_hgnn_embeddings",
    "fit_hypergcn_embeddings",
    "fit_hypergraph_embeddings",
    "get_hypergraph_embeddings",
    "train_unsupervised",
]
