# Unified HGNN and HyperGCN embeddings

Both methods use the same unsupervised graph-contrastive training pipeline and
the same public entry point.  Labels are not used during training.

```python
from src.HNN import HNNConfig, fit_hypergraph_embeddings

config = HNNConfig(
    embedding_dim=64,
    hidden_dim=64,
    num_layers=2,
    epochs=20,
    batch_size=32,
)

hgnn_result = fit_hypergraph_embeddings(
    ds,
    method="hgnn",
    config=config,
    feature_mode="auto",
    device="cpu",
)

hypergcn_result = fit_hypergraph_embeddings(
    ds,
    method="hypergcn",
    config=config,
    feature_mode="auto",
    device="cpu",
)

hgnn_embeddings = hgnn_result.embeddings
hypergcn_embeddings = hypergcn_result.embeddings
assert hgnn_result.hypergraph_ids == hypergcn_result.hypergraph_ids
```

For MUS, `feature_mode="auto"` selects shared node-identity plus degree
features because the global vocabulary has only 61 nodes.  Each result has
shape `(1944, embedding_dim)` and follows the order of
`ds.hypergraph_labels`.

HyperGCN uses native `torch.sparse.mm`; it does not require `torch_scatter` or
`torch_sparse`.  Use CPU or CUDA.  MPS is intentionally rejected because its
sparse-operation support is not reliable for this model.

To append node curvature as an initialization feature, pass one curvature
value per hyperedge.  For each node, the package averages finite curvatures of
its incident hyperedges and globally standardizes the resulting scalar before
appending it to the base node features. A node without any defined incident
curvature is assigned the global mean:

```python
hgnn_cihi = fit_hypergraph_embeddings(
    ds,
    method="hgnn",
    config=config,
    hyperedge_curvatures=cihi,
    device="cpu",
)
```
