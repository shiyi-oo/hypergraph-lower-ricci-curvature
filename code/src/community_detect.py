"""Hypergraph community detection algorithms.

Implemented methods:

- ``hypergraph_spectral_clustering`` follows Zhou, Huang, and Scholkopf's
  normalized hypergraph Laplacian ``I - Dv^-1/2 H W De^-1 H.T Dv^-1/2``.
- ``hypergraph_modularity_maximization`` is a Python implementation of the
  IRMM loop from https://github.com/tarunkumariitm/IRMM.
- ``hypergraph_clique_modularity_maximization`` is the one-shot clique Louvain
  baseline, useful for comparison with IRMM.
"""

import numpy as np
from typing import Optional

try:
    from .hg_class import HypergraphDataset      # imported as part of the `src` package
except ImportError:
    from hg_class import HypergraphDataset        # run/imported standalone from inside src/

# --------------------------------------------------------------------------- #
# hypergraph spectral clustering baseline (Zhou et al. NeurIPS 2006)          #
# --------------------------------------------------------------------------- #
def hypergraph_spectral_clustering(hg: HypergraphDataset, num_clusters: int,
                                   *, edge_weights=None, seed: int = 0,
                                   row_normalize: bool = False,
                                   solver: str = "auto",
                                   eig_tol: float = 1e-4,
                                   eig_max_iter: Optional[int] = None) -> dict:
    """Spectral clustering with the normalized hypergraph Laplacian.

    This is the k-way spectral embedding described in Zhou et al. (NeurIPS
    2006): use the eigenvectors associated with the ``num_clusters`` smallest
    eigenvalues of ``Delta = I - Dv^-1/2 H W De^-1 H.T Dv^-1/2`` as vertex
    features, then run k-means once. By default this uses a sparse partial
    eigensolver and avoids forming the dense Laplacian.

    ``solver`` can be ``"auto"``, ``"sparse"``, or ``"dense"``. Sparse mode
    computes the largest eigenvectors of
    ``Theta = Dv^-1/2 H W De^-1 H.T Dv^-1/2``, which are equivalent to the
    smallest eigenvectors of ``Delta = I - Theta``.

    Returns ``{node_id: cluster_label}``.
    """
    from sklearn.cluster import KMeans

    _validate_num_clusters(hg, num_clusters)
    nodes = hg.node_order()
    features = hypergraph_spectral_embedding(
        hg,
        num_clusters,
        edge_weights=edge_weights,
        solver=solver,
        eig_tol=eig_tol,
        eig_max_iter=eig_max_iter,
        row_normalize=row_normalize,
    )

    labels = KMeans(n_clusters=num_clusters, n_init=10,
                    random_state=seed).fit_predict(features)
    return dict(zip(nodes, labels.tolist()))


def hypergraph_spectral_embedding(hg: HypergraphDataset, num_clusters: int,
                                  *, edge_weights=None,
                                  row_normalize: bool = False,
                                  solver: str = "auto",
                                  eig_tol: float = 1e-4,
                                  eig_max_iter: Optional[int] = None) -> np.ndarray:
    """Return the Zhou-style spectral embedding for hypergraph nodes."""
    _validate_num_clusters(hg, num_clusters)
    features = _hypergraph_spectral_embedding(
        hg,
        num_clusters,
        edge_weights=edge_weights,
        solver=solver,
        eig_tol=eig_tol,
        eig_max_iter=eig_max_iter,
    )
    if row_normalize:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.where(norms == 0, 1.0, norms)
    return features

# --------------------------------------------------------------------------- #
# One shot Louvain modularity on the clique-expanded graph baseline           #
# --------------------------------------------------------------------------- #
def hypergraph_clique_modularity_maximization(hg: HypergraphDataset, *,
                                              weighted: bool = True,
                                              seed: int = 0) -> dict:
    """One-shot Louvain modularity on the clique-expanded graph baseline.

    This treats every hyperedge as a clique of equal-weight pairwise edges, so it
    is not the IRMM reweighted hypergraph method.

    Returns ``{node_id: cluster_label}``.
    """
    nodes = hg.node_order()
    A = hg.adjacency_matrix(weighted=weighted)
    communities = _louvain_communities(A, seed)
    return _communities_to_dict(communities, nodes)


# --------------------------------------------------------------------------- #
# IRMM: Iteratively Reweighted Modularity Maximization (Kumar et al.)          #
# --------------------------------------------------------------------------- #
def hypergraph_modularity_maximization(hg: HypergraphDataset,
                                       num_clusters: Optional[int] = None, *,
                                       max_iter: int = 12,
                                       damping: float = 0.5,
                                       seed: int = 0) -> dict:
    """Iteratively Reweighted Modularity Maximization for hypergraph clustering.

    This follows the MATLAB IRMM implementation at
    https://github.com/tarunkumariitm/IRMM. Each iteration:

    1. Build the weighted clique adjacency
       ``A_ij = sum_{e contains i,j} W(e) / (delta(e) - 1)`` with zero diagonal.
    2. Run Louvain modularity maximization on ``A``.
    3. Reweight hyperedges (``_update_weights``): hyperedges whose nodes sit in
       few clusters get upweighted, cut hyperedges downweighted.

    Parameters
    ----------
    hg : HypergraphDataset
    num_clusters : int, optional
        If given, the final partition is agglomeratively merged to exactly this
        many or fewer clusters (the MATLAB repo references a
        ``cluster_agglomorative_hypergraph`` step but does not include it).
        If None, the number of clusters is whatever Louvain finds.
    max_iter : int
        Number of reweighting iterations (repo uses 12; ``max_iter=1`` is plain
        weighted-clique "Hypergraph-Louvain", no reweighting feedback).
    damping : float
        Weight-update damping ``W <- damping*raw + (1-damping)*W_old`` (repo 0.5).
    seed : int
        Seed for Louvain.

    Returns
    -------
    dict
        ``{node_id: cluster_label}`` with contiguous 0-based labels.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if not 0.0 <= damping <= 1.0:
        raise ValueError("damping must be between 0 and 1")
    if num_clusters is not None:
        _validate_num_clusters(hg, num_clusters)

    nodes = hg.node_order()
    H = hg.get_incidence_matrix().astype(float)         # n x m, 0/1
    n, m = H.shape
    if n == 0:
        return {}

    de = H.sum(axis=0)                                  # delta(e), per hyperedge
    W = np.ones(m)

    labels = None
    for it in range(max_iter):
        A = hg.adjacency_matrix(edge_weights=W, normalize_by_edge_size=True)
        labels = _louvain_labels(A, n, seed)
        if num_clusters is not None and it == max_iter - 1:
            labels = _agglomerate_to_k(labels, A, num_clusters)
        if it != max_iter - 1:
            W = _update_weights(H, W, labels, de, damping)

    return dict(zip(nodes, labels.tolist()))


def _update_weights(H, W_old, labels, de, damping):
    """IRMM reweighting (port of HypergraphUtils.updateWeights).

    For each hyperedge e, with per-cluster node counts ``n_j`` over ALL k clusters
    (clusters not touched by e contribute 1/(0+1)=1):

        raw(e) = (delta(e) + k) * sum_j 1/(n_j + 1)

    then normalize so sum_e raw = m, and damp toward the old weights.
    """
    n, m = H.shape
    k = int(labels.max()) + 1
    onehot = np.zeros((n, k))
    onehot[np.arange(n), labels] = 1.0
    counts = H.T @ onehot                               # m x k : n_j(e)
    s = (1.0 / (counts + 1.0)).sum(axis=1)              # Σ_j 1/(n_j+1) over all k
    raw = (de + k) * s
    raw_sum = raw.sum()
    if raw_sum == 0:
        return W_old.copy()
    raw = m * raw / raw_sum
    return damping * raw + (1.0 - damping) * W_old


def _agglomerate_to_k(labels, A, k_target):
    """Greedily merge the two clusters with the largest inter-cluster weight in A
    until k_target clusters remain (faithful stand-in for the repo's missing
    ``cluster_agglomorative_hypergraph.m``)."""
    labels = labels.copy()
    while True:
        uniq = np.unique(labels)
        if len(uniq) <= k_target:
            break
        remap = {c: i for i, c in enumerate(uniq)}
        lab = np.array([remap[x] for x in labels])
        c = len(uniq)
        OH = np.zeros((len(labels), c))
        OH[np.arange(len(labels)), lab] = 1.0
        B = OH.T @ A @ OH                               # c x c inter-cluster weight
        np.fill_diagonal(B, -np.inf)                    # never "merge" with self
        a_, b_ = np.unravel_index(np.argmax(B), B.shape)
        labels[labels == uniq[b_]] = uniq[a_]           # merge b_ into a_
    return _relabel(labels)


def _validate_num_clusters(hg, num_clusters):
    n = hg.num_nodes
    if num_clusters < 1:
        raise ValueError("num_clusters must be at least 1")
    if n == 0:
        raise ValueError("cannot cluster an empty hypergraph")
    if num_clusters > n:
        raise ValueError("num_clusters cannot exceed the number of nodes")


def _hypergraph_spectral_embedding(hg, num_clusters, *, edge_weights, solver,
                                   eig_tol, eig_max_iter):
    n = hg.num_nodes
    if solver not in {"auto", "sparse", "dense"}:
        raise ValueError('solver must be "auto", "sparse", or "dense"')

    use_sparse = solver == "sparse" or (solver == "auto" and num_clusters < n - 1)
    if use_sparse:
        try:
            return _sparse_hypergraph_spectral_embedding(
                hg,
                num_clusters,
                edge_weights=edge_weights,
                eig_tol=eig_tol,
                eig_max_iter=eig_max_iter,
            )
        except Exception:
            if solver == "sparse":
                raise

    laplacian = hg.normalized_hypergraph_laplacian_matrix(edge_weights=edge_weights)
    _, eigenvectors = np.linalg.eigh(laplacian)       # symmetric -> ascending
    return eigenvectors[:, :num_clusters]


def _sparse_hypergraph_spectral_embedding(hg, num_clusters, *, edge_weights,
                                          eig_tol, eig_max_iter):
    from scipy.sparse.linalg import LinearOperator, eigsh

    H = hg.get_incidence_matrix_sparse()
    n, _ = H.shape
    weights = hg._edge_weight_array(edge_weights)

    vertex_degrees = np.asarray(H @ weights).ravel()
    edge_sizes = np.asarray(H.sum(axis=0)).ravel()

    inv_sqrt_dv = np.zeros_like(vertex_degrees, dtype=float)
    positive_vertices = vertex_degrees > 0
    inv_sqrt_dv[positive_vertices] = 1.0 / np.sqrt(vertex_degrees[positive_vertices])

    inv_de = np.zeros_like(edge_sizes, dtype=float)
    positive_edges = edge_sizes > 0
    inv_de[positive_edges] = 1.0 / edge_sizes[positive_edges]
    edge_scale = weights * inv_de
    HT = H.T.tocsr()

    def matvec(x):
        y = inv_sqrt_dv * x
        y = HT @ y
        y = edge_scale * y
        y = H @ y
        return inv_sqrt_dv * y

    theta = LinearOperator((n, n), matvec=matvec, dtype=float)
    eigenvalues, eigenvectors = eigsh(
        theta,
        k=num_clusters,
        which="LA",
        tol=eig_tol,
        maxiter=eig_max_iter,
    )
    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order]


def _louvain_communities(A, seed):
    import networkx as nx

    G = nx.from_numpy_array(A)
    if G.number_of_edges() == 0:
        return [{node} for node in G.nodes]
    if hasattr(nx.community, "louvain_communities"):
        return nx.community.louvain_communities(G, weight="weight", seed=seed)
    return nx.community.greedy_modularity_communities(G, weight="weight")


def _louvain_labels(A, n, seed):
    communities = _louvain_communities(A, seed)
    labels = np.empty(n, dtype=int)
    for ci, comm in enumerate(communities):
        for idx in comm:
            labels[idx] = ci
    return labels


def _relabel(labels):
    uniq = np.unique(labels)
    remap = {c: i for i, c in enumerate(uniq)}
    return np.array([remap[x] for x in labels])


def _communities_to_dict(communities, nodes):
    label_of = {}
    for cluster_id, comm in enumerate(communities):
        for idx in comm:
            label_of[nodes[idx]] = cluster_id
    return label_of
