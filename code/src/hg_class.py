import os
import pickle
import tempfile
from collections import Counter, defaultdict
from functools import cached_property
import numpy as np

class HypergraphDataset:
    """Canonical in-memory representation of a hypergraph dataset.

    A hypergraph is stored as ``hyperedges`` (a list of hyperedges, each a list
    of integer node ids).  Optional metadata:

    - ``node_labels``       : label per node, aligned to node id order.
    - ``edge_labels``       : label(s) per hyperedge, aligned to ``hyperedges``.
    - ``hypergraph_idx``    : id of the hypergraph each hyperedge belongs to
                              (for datasets that bundle many hypergraphs).
    - ``hypergraph_labels`` : label per hyperedge / hypergraph.

    Instances are persisted to ``data/processed/{dname}.pkl`` as a pickled
    Python class. CIHI, HFRC, HORC, and BE are computed lazily and written back
    to that pickle so later loads can reuse them.
    """

    def __init__(self, hyperedges, node_labels=None, edge_labels=None,
                 hypergraph_idx=None, hypergraph_labels=None):
        self.hyperedges = hyperedges
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.hypergraph_idx = hypergraph_idx
        self.hypergraph_labels = hypergraph_labels

    # Structural cached properties are cheap to rebuild and are excluded from
    # the pickle. Curvature values (cihi, hfrc, horc_all, be) remain in the pickle
    # because they can be expensive and are intended as persistent caches.
    _TRANSIENT_FIELDS = (
        "node_degrees",
        "edge_degrees",
        "node_neighbors",
        "edge_neighborhood",
        "_cache_path",
        "horc",
    )

    def __getstate__(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in self._TRANSIENT_FIELDS
        }

    @property
    def num_hyperedges(self) -> int:
        return len(self.hyperedges)

    @property
    def num_nodes(self) -> int:
        return len({node for edge in self.hyperedges for node in edge})

    def __len__(self):
        return len(self.hyperedges)

    def __repr__(self):
        edge_degrees = self.edge_degrees
        node_degrees = self.node_degrees
        num_hyperedges = self.num_hyperedges
        num_deg2 = edge_degrees.get(2, 0)
        pct_deg2 = (num_deg2 / num_hyperedges) * 100 if num_hyperedges > 0 else 0.0
        num_hypergraphs = len(set(self.hypergraph_idx)) if self.hypergraph_idx else 1
        max_dv = max(node_degrees.values()) if node_degrees else 0
        max_de = max((len(edge) for edge in self.hyperedges), default=0)

        lines = [
            f"HypergraphDataset(num_hyperedges={num_hyperedges}, "
            f"num_nodes={self.num_nodes}, "
            f"num_hypergraphs={num_hypergraphs}, "
            f"node_labels={'yes' if self.node_labels is not None else 'no'}, "
            f"edge_labels={'yes' if self.edge_labels is not None else 'no'}, "
            f"hypergraph_labels={'yes' if self.hypergraph_labels is not None else 'no'}, "
            f"max_dv={max_dv}, max_de={max_de}, perc_de2={pct_deg2:.1f}%)"
        ]
        def uniq(labels, limit=10):
            # dedup over ALL labels; tuple-ify unhashable (e.g. [year, journal]) entries
            u = {tuple(x) if isinstance(x, list) else x for x in labels}
            return len(u), sorted(u)[:limit]

        if self.node_labels is not None:
            n, sample = uniq(self.node_labels)
            lines.append(f"Unique Node labels ({n}): {sample}")
        if self.edge_labels is not None:
            n, sample = uniq(self.edge_labels)
            lines.append(f"Unique Edge labels ({n}): {sample}")
        if self.hypergraph_labels is not None:
            n, sample = uniq(self.hypergraph_labels)
            lines.append(f"Unique Hypergraph labels ({n}): {sample}")
        return "\n".join(lines)

    def save(self, path):
        """Atomically pickle this dataset and use it as the metric cache."""
        path = os.path.abspath(os.fspath(path))
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            dir=parent or ".",
        )
        try:
            with os.fdopen(descriptor, "wb") as file:
                pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_path, path)
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise

        self._cache_path = path

    @classmethod
    def load(cls, path):
        """Unpickle a :class:`HypergraphDataset` from ``path``.

        Uses a compatibility unpickler so a cache stays loadable no matter which
        name this module was imported under when it was saved (e.g. top-level
        ``hg_class`` when run as a script vs ``src.hg_class`` as a package).
        """
        path = os.path.abspath(os.fspath(path))
        with open(path, "rb") as f:
            obj = _CompatUnpickler(f).load()
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        obj._cache_path = os.fspath(path)
        return obj

    def _persist_metric_cache(self):
        """Write derived metrics back when this instance came from a cache."""
        cache_path = getattr(self, "_cache_path", None)
        if cache_path is not None:
            self.save(cache_path)

    @cached_property
    def node_degrees(self) -> Counter:
        """
        Degree of each node (i.e. in how many hyperedges it appears).
        Computed on first access and cached on the instance.
        """
        return Counter(node for edge in self.hyperedges for node in edge)

    @cached_property
    def edge_degrees(self) -> Counter:
        """
        Size of each hyperedge as a Counter of {edge_size: count}.
        Computed on first access and cached on the instance.
        """
        return Counter(len(edge) for edge in self.hyperedges)

    @cached_property
    def node_neighbors(self) -> dict[int, set[int]]:
        """
        For each node, the set of all other nodes with which it co-occurs in any hyperedge.
        Computed on first access and cached on the instance.
        """
        v_neigh: dict[int, set[int]] = defaultdict(set)
        for edge in self.hyperedges:
            s = set(edge)
            for v in edge:
                v_neigh[v].update(s - {v})
        return dict(v_neigh)

    @cached_property
    def edge_neighborhood(self) -> dict[int, set[int]]:
        """
        For each hyperedge (by its index in H), the intersection of its member-nodes' neighborhoods.
        Computed on first access and cached on the instance.
        """
        v_neigh = self.node_neighbors
        e_neigh: dict[int, set[int]] = defaultdict(set)
        for idx, edge in enumerate(self.hyperedges):
            if len(edge) <= 1:
                e_neigh[idx] = set()
            else:
                # intersection of neighbors[v] for v in edge
                e_neigh[idx] = set.intersection(*(v_neigh.get(v, set()) for v in edge))
        return e_neigh
    
    def get_incidence_matrix(self) -> np.ndarray:
        """
        Incidence matrix (nodes x hyperedges) as a NumPy array:
        rows = all nodes, sorted
        columns = hyperedge indices (0,1,2,…)
        entry is 1 if node ∈ edge, else 0
        """
        all_nodes = sorted({n for edge in self.hyperedges for n in edge})  # sorted list of nodes
        node_index = {n: i for i, n in enumerate(all_nodes)}
        M = np.zeros((len(all_nodes), len(self.hyperedges)), dtype=int)

        for e_idx, edge in enumerate(self.hyperedges):
            for n in edge:
                M[node_index[n], e_idx] = 1

        return M

    def get_incidence_matrix_sparse(self):
        """Sparse CSR incidence matrix with rows in ``node_order()``."""
        from scipy import sparse

        all_nodes = self.node_order()
        node_index = {n: i for i, n in enumerate(all_nodes)}

        rows = []
        cols = []
        for e_idx, edge in enumerate(self.hyperedges):
            for n in set(edge):
                rows.append(node_index[n])
                cols.append(e_idx)

        data = np.ones(len(rows), dtype=float)
        shape = (len(all_nodes), len(self.hyperedges))
        return sparse.coo_matrix((data, (rows, cols)), shape=shape).tocsr()

    def node_order(self) -> list[int]:
        """Sorted unique node ids — the row/column order of the matrices below."""
        return sorted({n for edge in self.hyperedges for n in edge})

    def _edge_weight_array(self, edge_weights=None) -> np.ndarray:
        """Return validated hyperedge weights aligned with ``self.hyperedges``."""
        if edge_weights is None:
            return np.ones(len(self.hyperedges), dtype=float)

        weights = np.asarray(edge_weights, dtype=float)
        if weights.shape != (len(self.hyperedges),):
            raise ValueError(
                "edge_weights must have length equal to num_hyperedges "
                f"({len(self.hyperedges)})"
            )
        if np.any(weights < 0):
            raise ValueError("edge_weights must be nonnegative")
        return weights

    def _hyperedge_blocks(self):
        """Yield hyperedge indices grouped by hypergraph id, preserving row order."""
        if self.hypergraph_idx is None:
            yield list(range(len(self.hyperedges)))
            return

        if len(self.hypergraph_idx) != len(self.hyperedges):
            raise ValueError(
                "hypergraph_idx must have length equal to num_hyperedges "
                f"({len(self.hyperedges)})"
            )

        blocks = defaultdict(list)
        for edge_idx, hg_idx in enumerate(self.hypergraph_idx):
            blocks[hg_idx].append(edge_idx)

        yield from blocks.values()

    @staticmethod
    def _node_degrees_for_edges(edges) -> Counter:
        return Counter(node for edge in edges for node in edge)

    @staticmethod
    def _node_neighbors_for_edges(edges) -> dict[int, set[int]]:
        v_neigh: dict[int, set[int]] = defaultdict(set)
        for edge in edges:
            s = set(edge)
            for v in edge:
                v_neigh[v].update(s - {v})
        return dict(v_neigh)

    def hyperedge_degree_vector(self) -> np.ndarray:
        """Hyperedge sizes ``delta(e)`` in incidence-matrix column order."""
        H = self.get_incidence_matrix().astype(float)
        return H.sum(axis=0)

    def vertex_degree_vector(self, edge_weights=None) -> np.ndarray:
        """Weighted vertex degrees ``d(v) = sum_{e contains v} w(e)``."""
        H = self.get_incidence_matrix().astype(float)
        weights = self._edge_weight_array(edge_weights)
        return H @ weights

    def adjacency_matrix(self, weighted: bool = True, edge_weights=None,
                         normalize_by_edge_size: bool = False) -> np.ndarray:
        """Clique-expansion adjacency matrix (n_nodes x n_nodes), rows/cols in
        ``node_order()``.

        ``A[i, j]`` is the total weight of hyperedges containing both node i and
        node j, with a zero diagonal. With ``weighted=False`` the entries are
        converted to 0/1 after aggregation.

        If ``normalize_by_edge_size=True``, hyperedge ``e`` contributes
        ``w(e) / (|e| - 1)`` to each off-diagonal pair in the hyperedge. This is
        the weighted clique expansion used by IRMM-style modularity
        maximization. Singleton hyperedges contribute no pairwise edges.
        """
        H = self.get_incidence_matrix().astype(float)   # n_nodes x n_edges, 0/1
        weights = self._edge_weight_array(edge_weights)
        if normalize_by_edge_size:
            edge_sizes = H.sum(axis=0)
            scale = np.zeros_like(edge_sizes, dtype=float)
            pairwise_edges = edge_sizes > 1
            scale[pairwise_edges] = 1.0 / (edge_sizes[pairwise_edges] - 1.0)
            weights = weights * scale

        A = (H * weights) @ H.T
        np.fill_diagonal(A, 0.0)
        if not weighted:
            A = (A > 0).astype(float)
        return A

    def degree_matrix(self, weighted: bool = True, edge_weights=None,
                      normalize_by_edge_size: bool = False) -> np.ndarray:
        """Diagonal degree matrix of the clique-expansion graph: ``D[i, i]`` is the
        (weighted) degree of node i, i.e. the row sum of ``adjacency_matrix()``.

        Defined so that ``degree_matrix() - adjacency_matrix()`` is a valid graph
        Laplacian (rows sum to zero).
        """
        A = self.adjacency_matrix(
            weighted=weighted,
            edge_weights=edge_weights,
            normalize_by_edge_size=normalize_by_edge_size,
        )
        return np.diag(A.sum(axis=1))

    def normalized_hypergraph_laplacian_matrix(self, edge_weights=None) -> np.ndarray:
        """Zhou-Huang-Scholkopf normalized hypergraph Laplacian.

        Returns ``Delta = I - Theta`` where
        ``Theta = Dv^-1/2 H W De^-1 H.T Dv^-1/2``. Here ``Dv`` contains weighted
        vertex degrees ``sum_e w(e) h(v,e)`` and ``De`` contains hyperedge sizes
        ``delta(e)``.
        """
        H = self.get_incidence_matrix().astype(float)
        n = H.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=float)

        weights = self._edge_weight_array(edge_weights)
        vertex_degrees = H @ weights
        edge_sizes = H.sum(axis=0)

        inv_sqrt_dv = np.zeros_like(vertex_degrees, dtype=float)
        positive_vertices = vertex_degrees > 0
        inv_sqrt_dv[positive_vertices] = 1.0 / np.sqrt(vertex_degrees[positive_vertices])

        inv_de = np.zeros_like(edge_sizes, dtype=float)
        positive_edges = edge_sizes > 0
        inv_de[positive_edges] = 1.0 / edge_sizes[positive_edges]

        theta = (H * (weights * inv_de)) @ H.T
        theta = inv_sqrt_dv[:, None] * theta * inv_sqrt_dv[None, :]

        laplacian = np.eye(n) - theta
        zero_degree = ~positive_vertices
        laplacian[zero_degree, :] = 0.0
        laplacian[:, zero_degree] = 0.0
        return laplacian

    def get_hfrc(self, *, recompute=False):
        """
        Compute H-FRC curvature for each hyperedge.

        If this dataset bundles multiple hypergraphs via ``hypergraph_idx``,
        node degrees are computed separately inside each hypergraph.
        """
        cached = getattr(self, "hfrc", None)
        if (
            not recompute
            and cached is not None
            and len(cached) == len(self.hyperedges)
        ):
            return cached

        hfrc = [None] * len(self.hyperedges)
        for block in self._hyperedge_blocks():
            edges = [self.hyperedges[i] for i in block]
            Dv = self._node_degrees_for_edges(edges)
            for edge_idx, edge in zip(block, edges):
                d_e = len(edge)
                deg_sum = sum(Dv[v] for v in edge)
                hfrc[edge_idx] = 2 * d_e - deg_sum

        self.hfrc = hfrc
        self._persist_metric_cache()
        return hfrc

    def get_be(self, *, dimension=float("inf"), recompute=False):
        """Compute Bakry--Émery curvature for each hyperedge.

        Each hyperedge is treated as a node in the hypergraph line graph.  The
        line graph is unweighted: two hyperedge-nodes are adjacent when the
        corresponding hyperedges intersect.  Unnormalised, unweighted
        Bakry--Émery curvature is then computed on that line graph.

        If this dataset bundles multiple hypergraphs via ``hypergraph_idx``,
        each hypergraph is converted to a line graph independently.
        """
        cached = getattr(self, "be", None)
        if (
            not recompute
            and cached is not None
            and len(cached) == len(self.hyperedges)
        ):
            return cached

        if not self.hyperedges:
            self.be = []
            self._persist_metric_cache()
            return self.be

        import networkx as nx
        import xgi

        from .be import non_normalised_unweighted_curvature

        be = [None] * len(self.hyperedges)
        for block in self._hyperedge_blocks():
            if not block:
                continue

            hypergraph = xgi.Hypergraph(
                {edge_idx: self.hyperedges[edge_idx] for edge_idx in block}
            )
            line_graph = xgi.to_line_graph(hypergraph)
            line_graph.add_nodes_from(block)
            adjacency = nx.to_numpy_array(
                line_graph,
                nodelist=block,
                dtype=float,
            )
            block_values = non_normalised_unweighted_curvature(
                adjacency,
                dimension,
            )
            for edge_idx, value in zip(block, block_values):
                be[edge_idx] = value

        self.be = be
        self._persist_metric_cache()
        return be

    def get_horc(
        self,
        dispersion="uw_clique",
        aggregation="mean",
        alpha=0.0,
        *,
        all_configurations=False,
        julia=None,
        threads="auto",
        recompute=False,
    ):
        """Compute ORCHID hyperedge Ollivier--Ricci curvature (HORC).

        Parameters
        ----------
        dispersion : {"uw_clique", "w_clique", "uw_star"}
            Probability-dispersion rule.  The default is the unweighted
            clique rule used by the legacy ``horc`` column.
        aggregation : {"mean", "max"}
            Pairwise transport-cost aggregation within each hyperedge.
        alpha : {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
            Self-dispersion weight.
        all_configurations : bool, default=False
            Return all 36 configurations as ``{column_name: values}`` instead
            of returning one curvature list.
        julia : str or None
            Julia executable.  ``None`` resolves ``julia`` from ``PATH``.
        threads : str or int, default="auto"
            Value passed to Julia's ``--threads`` option.
        recompute : bool, default=False
            Ignore cached HORC values and run Orchid again.

        Returns
        -------
        list[float] or dict[str, list[float]]
            Curvatures aligned with ``self.hyperedges``.  If
            ``self.hypergraph_idx`` is present, each hypergraph is computed
            independently.

        Notes
        -----
        HORC is implemented by the repository's Julia ``Orchid`` project.
        The first call computes and caches all 36 configurations so subsequent
        configuration lookups do not rerun optimal transport.
        """
        import shutil
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import pandas as pd

        from .horc import (
            AGGREGATIONS,
            ALPHAS,
            DISPERSIONS,
            HORC_COLUMNS,
            compute_horc,
            horc_column_name,
        )

        if dispersion not in DISPERSIONS:
            choices = ", ".join(DISPERSIONS)
            raise ValueError(f"dispersion must be one of: {choices}")
        if aggregation not in AGGREGATIONS:
            choices = ", ".join(AGGREGATIONS)
            raise ValueError(f"aggregation must be one of: {choices}")

        alpha = float(alpha)
        if alpha not in ALPHAS:
            choices = ", ".join(map(str, ALPHAS))
            raise ValueError(f"alpha must be one of: {choices}")

        selected_column = horc_column_name(dispersion, aggregation, alpha)
        cached = getattr(self, "horc_all", None)
        cache_is_valid = (
            isinstance(cached, dict)
            and all(
                column in cached
                and len(cached[column]) == len(self.hyperedges)
                for column in HORC_COLUMNS
            )
        )
        cache_updated = recompute or not cache_is_valid
        if cache_updated:
            if not self.hyperedges:
                cached = {column: [] for column in HORC_COLUMNS}
            else:
                # Map arbitrary hashable node IDs to collision-free strings;
                # HORC depends only on incidence, not on node names.
                node_ids = {}
                for edge in self.hyperedges:
                    for node in edge:
                        if node not in node_ids:
                            node_ids[node] = str(len(node_ids))

                hypergraph_ids = [None] * len(self.hyperedges)
                for group_idx, edge_indices in enumerate(self._hyperedge_blocks()):
                    for edge_idx in edge_indices:
                        hypergraph_ids[edge_idx] = group_idx

                members = [
                    [node_ids[node] for node in edge]
                    for edge in self.hyperedges
                ]

                with TemporaryDirectory() as directory:
                    directory_path = Path(directory)
                    source_path = directory_path / "hypergraphs.csv"
                    output_path = directory_path / "hypergraphs_horc.csv"
                    pd.DataFrame(
                        {
                            "hg_idx": hypergraph_ids,
                            "config_idx": 0,
                            "members": members,
                        }
                    ).to_csv(source_path, index=False)

                    julia_executable = julia or shutil.which("julia") or "julia"
                    compute_horc(
                        source_path,
                        output_path,
                        julia=julia_executable,
                        threads=str(threads),
                    )
                    result = pd.read_csv(output_path, usecols=list(HORC_COLUMNS))

                cached = {
                    column: result[column].astype(float).tolist()
                    for column in HORC_COLUMNS
                }

            self.horc_all = cached

        self.horc = cached[selected_column]
        if cache_updated:
            self._persist_metric_cache()
        return cached if all_configurations else self.horc

    def get_cihi(self, *, recompute=False):
        """
        Compute CIHI for each hyperedge.

        If this dataset bundles multiple hypergraphs via ``hypergraph_idx``,
        node neighborhoods are computed separately inside each hypergraph.
        """
        cached = getattr(self, "cihi", None)
        if (
            not recompute
            and cached is not None
            and len(cached) == len(self.hyperedges)
        ):
            return cached

        cihi = [None] * len(self.hyperedges)
        for block in self._hyperedge_blocks():
            edges = [self.hyperedges[i] for i in block]
            v_neigh = self._node_neighbors_for_edges(edges)

            for edge_idx, edge in zip(block, edges):
                d_e = len(set(edge))
                if d_e <= 1:
                    continue

                neigh_sizes = [len(v_neigh[v]) for v in edge]
                max_size, min_size = max(neigh_sizes), min(neigh_sizes)
                if max_size == 0 or min_size == 0:
                    continue

                common = set.intersection(*(v_neigh[v] for v in edge))
                n_e = len(common)
                sum_recip = sum(1 / s for s in neigh_sizes)

                cihi[edge_idx] = (
                    sum_recip - 1
                    + (n_e + d_e/2 - 1) / max_size
                    + (n_e + d_e/2 - 1) / min_size
                )

        self.cihi = cihi
        self._persist_metric_cache()
        return cihi


class _CompatUnpickler(pickle.Unpickler):
    """Resolve HypergraphDataset regardless of the module name stored in the
    pickle, so caches written under ``hg_class`` / ``src.hg_class`` / ``__main__``
    all load against the current class."""

    def find_class(self, module, name):
        if name == "HypergraphDataset":
            return HypergraphDataset
        return super().find_class(module, name)


if __name__ == "__main__":
    # Example usage
    hyperedges = [[1, 2], [2, 3], [1, 3, 4]]
    dataset = HypergraphDataset(hyperedges)
    print(dataset)
    # HypergraphDataset(num_hyperedges=3, num_nodes=4, num_hypergraphs=1, node_labels=no, edge_labels=no, hypergraph_labels=no, max_dv=2, max_de=3, perc_de2=66.7%)
