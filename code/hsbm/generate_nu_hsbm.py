"""Generate synthetic hypergraphs and compare curvature notions.

For each parameter configuration in ``PARAM_SETS`` the script:
    * Synthesises a hypergraph using ``nu_hsbm.Generator``
    * Computes HLRC, HLRC_naive, HFRC, and BE curvature scores per hyperedge
    * Appends rows to a combined summary CSV spanning all runs
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import networkx as nx
import xgi
import pandas as pd
import random

import sys
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
PROJECT_STR = str(PROJECT_ROOT)
if PROJECT_STR not in sys.path:
    sys.path.append(PROJECT_STR)

from src.hlrc import compute_hlrc, compute_hlrc_naive
from src.hfrc import compute_hfrc
from src.be import inf, normalised_unweighted_curvature
from src.nu_hsbm import Generator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "derived_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ParamSet:
    """Container describing a synthetic hypergraph experiment."""
    num_nodes: int
    num_edges: int
    num_comms: int
    p_intra: float
    p_inter: float
    inter_to_intra_ratio: float
    sampling_strat: str

    def describe(self) -> str:
        return (
            "Non-Uniform Stochastic Block Hypergraph Model: "
            f"{self.num_comms} communities, {self.num_nodes} nodes, {self.num_edges} hyperedges; "
            f"p_intra={self.p_intra}, (q/p={self.inter_to_intra_ratio})"
        )


RATIO_VALUES = [0.05, 0.1, 0.25, 0.5, 0.75]
P_INTRA_VALUES = [0.10]

NODES_PER_COMMUNITY = 20
EDGES_PER_COMMUNITY = 60


def build_param_sets() -> List[ParamSet]:
    """Enumerate parameter combinations covering communities 2 - 5 and q/p ratios."""

    param_sets: List[ParamSet] = []

    for num_comms in (2, 3, 4):
        num_nodes = num_comms * NODES_PER_COMMUNITY
        num_edges = num_comms * EDGES_PER_COMMUNITY

        for p_intra in P_INTRA_VALUES:
            for ratio in RATIO_VALUES:
                p_inter = p_intra * ratio
                param_sets.append(
                    ParamSet(
                        num_nodes=num_nodes,
                        num_edges=num_edges,
                        num_comms=num_comms,
                        p_intra=p_intra,
                        p_inter=p_inter,
                        sampling_strat="frequent",
                        inter_to_intra_ratio=ratio,
                    )
                )

    return param_sets


PARAM_SETS: List[ParamSet] = build_param_sets()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_metrics(params: ParamSet) -> pd.DataFrame:
    """Build a hypergraph for the given parameters and compute curvature values."""

    gen = Generator(
        n_nodes=params.num_nodes,
        n_hedges=params.num_edges,
        n_coms=params.num_comms,
        p_edge_intra=params.p_intra,
        p_edge_inter=params.p_inter,
        sampling_strat=params.sampling_strat,
    )
    gen.run()

    hyperedges = {
        hedge: sorted(gen.G.neighbors(hedge))
        for hedge, attr in gen.G.nodes(data=True)
        if attr["type"] == gen.hyperedge_type
    }

    node_communities = {
        node: attr["community"]
        for node, attr in gen.G.nodes(data=True)
        if attr["type"] == gen.node_type
    }

    he_labels = {
        hedge: (
            "intra"
            if len({node_communities[n] for n in members}) == 1
            else "inter"
        )
        for hedge, members in hyperedges.items()
    }

    ordered_edges: List[str] = list(hyperedges.keys())
    edge_members: List[List[str]] = [hyperedges[hedge] for hedge in ordered_edges]

    hlrc = compute_hlrc(edge_members)
    hlrc_naive = compute_hlrc_naive(edge_members)
    hfrc = compute_hfrc(edge_members)

    # be = None
    # H = xgi.Hypergraph(hyperedges)
    # line_graph = xgi.to_line_graph(H)
    # adjacency = nx.to_pandas_adjacency(line_graph)
    # be = normalised_unweighted_curvature(adjacency, inf)

    df = pd.DataFrame(
        {
            "edge": ordered_edges,
            "members": edge_members,
            "edge_label": [he_labels[e] for e in ordered_edges],
            "hlrc": hlrc,
            "hfrc": hfrc,
            # "be": be,
            "hlrc_naive": hlrc_naive,
            "num_nodes": params.num_nodes,
            "num_edges": params.num_edges,
            "num_comms": params.num_comms,
            "p_intra": params.p_intra,
            "p_inter": params.p_inter,
            "inter_to_intra_ratio": params.inter_to_intra_ratio,
        }
    )

    return df

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    REPLICATES = 100
    combined_frames = []
    hg_idx = 0
    for config_idx, params in enumerate(PARAM_SETS):
        for rep in range(REPLICATES):
            seed = hg_idx * 10 + rep
            random.seed(seed)
            df = generate_metrics(params)
            df.insert(0, "config_idx", config_idx)
            df.insert(1, "hg_idx", hg_idx)
            hg_idx += 1
            combined_frames.append(df)

    if combined_frames:
        combined_df = pd.concat(combined_frames, ignore_index=True)
        combined_csv = DATA_DIR / "nu_hsbm_dataset.csv"
        combined_df.to_csv(combined_csv, index=False)


if __name__ == "__main__":
    main()
