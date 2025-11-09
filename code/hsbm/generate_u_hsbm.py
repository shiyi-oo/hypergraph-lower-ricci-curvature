from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xgi
import networkx as nx

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
PROJECT_STR = str(PROJECT_ROOT)
if PROJECT_STR not in sys.path:
    sys.path.append(PROJECT_STR)

from src.hlrc import compute_hlrc
from src.hfrc import compute_hfrc
from src.be import inf, normalised_unweighted_curvature
from src.hsbm import generate_hsbm

PARAM_SETS: list[dict[str, Any]] = [
    {
        "name": "k3_two_comm",
        "k_uniform": 3,
        "block_sizes": [15, 15],
        "community_edge_probs": [0.1, 0.001],
        "seed": 123,
    },
    {
        "name": "k3_two_comm_unbalanced",
        "k_uniform": 3,
        "block_sizes": [15, 25],
        "community_edge_probs": [0.1, 0.001],
        "seed": 123,
    },
    {
        "name": "k3_three_comm",
        "k_uniform": 3,
        "block_sizes": [15, 15, 15],
        "community_edge_probs": [0.1, 0.001, 0.001],
        "seed": 123,
    },
    {
        "name": "k3_three_comm_unbalanced",
        "k_uniform": 3,
        "block_sizes": [40, 30, 20],
        "community_edge_probs": [0.1, 0.001, 0.001],
        "seed": 123,
    },
    {
        "name": "k4_two_comm",
        "k_uniform": 4,
        "block_sizes": [15, 15],
        "community_edge_probs": [0.1, 0.001],
        "seed": 123,
    },
    {
        "name": "k4_two_comm_unbalanced",
        "k_uniform": 4,
        "block_sizes": [20, 40],
        "community_edge_probs": [1e-2, 1e-4],
        "seed": 123,
    },
    {
        "name": "k4_three_comm_balanced",
        "k_uniform": 4,
        "block_sizes": [15, 15, 15],
        "community_edge_probs": [0.1, 0.001, 0.001],
        "seed": 123,
    },
    {
        "name": "k4_three_comm_unbalanced",
        "k_uniform": 4,
        "block_sizes": [40, 30, 20],
        "community_edge_probs": [1e-2, 1e-4, 1e-4],
        "seed": 123,
    },
]

def _edge_label(num_communities: int) -> str:
    """Label hyperedges as intra vs inter community."""
    return "inter" if num_communities > 1 else "intra"


def build_dataset(param_sets: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Generate HSBM samples for all parameter sets and aggregate into one DataFrame."""
    frames: list[pd.DataFrame] = []

    for idx, raw_params in enumerate(param_sets):
        params = dict(raw_params)
        name = params.pop("name", f"config_{idx}")
        seed = params.setdefault("seed", 123)

        hyperedges, node_labels, communities_per_edge = generate_hsbm(**params)
        hyperedges = [[str(n) for n in edge] for edge in hyperedges]

        hlrc_values = compute_hlrc(hyperedges)
        hfrc_values = compute_hfrc(hyperedges)
        frame = pd.DataFrame(
            {
                "hg_idx": idx,
                "config_name": name,
                "edge": range(len(hyperedges)),
                "members": hyperedges,
                "hlrc": hlrc_values,
                "hfrc": hfrc_values,
                "edge_label": [_edge_label(n) for n in communities_per_edge],
                "k_uniform": params["k_uniform"],
                "num_comms": len(params["block_sizes"]),
                "block_sizes": "|".join(map(str, params["block_sizes"])),
                "community_edge_probs": "|".join(map(str, params["community_edge_probs"])),
                "seed": seed,
                "node_labels": [node_labels] * len(hyperedges),
            }
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def main(output_path: Path | None = None) -> Path:
    """Generate the dataset and write it to disk."""
    if output_path is None:
        output_path = CURRENT_DIR / "derived_data" / "u_hsbm_toy_dataset.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset(PARAM_SETS)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    saved_path = main()
    print(f"U-HSBM dataset written to {saved_path}")
