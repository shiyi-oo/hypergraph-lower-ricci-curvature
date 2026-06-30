"""Generate the uniform and non-uniform HSBM simulation datasets."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = CURRENT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.hg_class import HypergraphDataset
from src.hsbm import generate_nonuniform_hsbm, generate_uniform_hsbm

DERIVED_DIR = CURRENT_DIR / "derived"
UNIFORM_OUTPUT = DERIVED_DIR / "u_hsbm_dataset.csv"
NONUNIFORM_OUTPUT = DERIVED_DIR / "nu_hsbm_dataset.csv"


@dataclass(frozen=True)
class UniformConfig:
    """Configuration for one uniform HSBM sample."""

    name: str
    k_uniform: int
    block_sizes: tuple[int, ...]
    community_edge_probs: tuple[float, ...]
    seed: int = 123


UNIFORM_CONFIGS = (
    UniformConfig("k3_two_comm", 3, (15, 15), (0.1, 0.001)),
    UniformConfig("k3_two_comm_unbalanced", 3, (15, 25), (0.1, 0.001)),
    UniformConfig("k3_three_comm", 3, (15, 15, 15), (0.1, 0.001, 0.001)),
    UniformConfig("k3_three_comm_unbalanced",3,(40, 30, 20),(0.1, 0.001, 0.001),),
    UniformConfig("k4_two_comm", 4, (15, 15), (0.1, 0.001)),
    UniformConfig("k4_two_comm_unbalanced", 4, (20, 40), (1e-2, 1e-4)),
    UniformConfig("k4_three_comm_balanced",4,(15, 15, 15),(0.1, 0.001, 0.001),),
    UniformConfig("k4_three_comm_unbalanced",4,(40, 30, 20),(1e-2, 1e-4, 1e-4),),
)


@dataclass(frozen=True)
class NonuniformConfig:
    """Configuration for one non-uniform HSBM experiment."""

    num_nodes: int
    num_edges: int
    num_comms: int
    p_intra: float
    p_inter: float
    inter_to_intra_ratio: float
    sampling_strategy: str = "frequent"


NONUNIFORM_RATIO_VALUES = (0.05, 0.1, 0.25, 0.5, 0.75)
NONUNIFORM_P_INTRA_VALUES = (0.10,)
NODES_PER_COMMUNITY = 20
EDGES_PER_COMMUNITY = 60
NONUNIFORM_REPLICATES = 100


def build_nonuniform_configs() -> tuple[NonuniformConfig, ...]:
    """Return the 15 configurations from the old non-uniform script."""
    configs: list[NonuniformConfig] = []

    for num_comms in (2, 3, 4):
        for p_intra in NONUNIFORM_P_INTRA_VALUES:
            for ratio in NONUNIFORM_RATIO_VALUES:
                configs.append(
                    NonuniformConfig(
                        num_nodes=num_comms * NODES_PER_COMMUNITY,
                        num_edges=num_comms * EDGES_PER_COMMUNITY,
                        num_comms=num_comms,
                        p_intra=p_intra,
                        p_inter=p_intra * ratio,
                        inter_to_intra_ratio=ratio,
                    )
                )

    return tuple(configs)


NONUNIFORM_CONFIGS = build_nonuniform_configs()


def build_uniform_dataset(
    configs: Iterable[UniformConfig] = UNIFORM_CONFIGS,
) -> pd.DataFrame:
    """Generate all uniform HSBM configurations and their curvature values."""
    frames: list[pd.DataFrame] = []

    for hg_idx, config in enumerate(configs):
        hyperedges, node_labels, community_counts = generate_uniform_hsbm(
            k_uniform=config.k_uniform,
            block_sizes=config.block_sizes,
            community_edge_probs=config.community_edge_probs,
            seed=config.seed,
        )
        members = [edge.copy() for edge in hyperedges]
        dataset = HypergraphDataset(members)

        frame = pd.DataFrame(
            {
                "hg_idx": hg_idx,
                "config_name": config.name,
                "edge": range(len(members)),
                "members": members,
                "hlrc": dataset.get_cihi(),
                "hfrc": dataset.get_hfrc(),
                "edge_label": [_edge_label(count) for count in community_counts],
                "k_uniform": config.k_uniform,
                "num_comms": len(config.block_sizes),
                "block_sizes": "|".join(map(str, config.block_sizes)),
                "community_edge_probs": "|".join(
                    map(str, config.community_edge_probs)
                ),
                "seed": config.seed,
                "node_labels": [node_labels] * len(members),
            }
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_nonuniform_dataset(
    configs: Sequence[NonuniformConfig] = NONUNIFORM_CONFIGS,
    replicates: int = NONUNIFORM_REPLICATES,
    verbose: bool = False,
) -> pd.DataFrame:
    """Generate the old sequential-growth non-uniform HSBM dataset."""
    frames: list[pd.DataFrame] = []
    hg_idx = 0

    for config_idx, config in enumerate(configs):
        for replicate in range(replicates):
            # Preserve the seed schedule in generate_nu_hsbm.py exactly.
            seed = hg_idx * 10 + replicate
            hyperedges, _, community_counts = generate_nonuniform_hsbm(
                n_hyperedges=config.num_edges,
                block_sizes=[NODES_PER_COMMUNITY] * config.num_comms,
                p_edge_intra=config.p_intra,
                p_edge_inter=config.p_inter,
                sampling_strategy=config.sampling_strategy,
                order_strategy="random",
                seed=seed,
            )

            edge_ids = [f"h{edge_idx}" for edge_idx in range(len(hyperedges))]
            members = [
                sorted(f"p{node}" for node in edge)
                for edge in hyperedges
            ]
            dataset = HypergraphDataset(members)

            frame = pd.DataFrame(
                {
                    "config_idx": config_idx,
                    "hg_idx": hg_idx,
                    "edge": edge_ids,
                    "members": members,
                    "edge_label": [
                        _edge_label(count) for count in community_counts
                    ],
                    "hlrc": dataset.get_cihi(),
                    "hfrc": dataset.get_hfrc(),
                    "num_nodes": config.num_nodes,
                    "num_edges": config.num_edges,
                    "num_comms": config.num_comms,
                    "p_intra": config.p_intra,
                    "p_inter": config.p_inter,
                    "inter_to_intra_ratio": config.inter_to_intra_ratio,
                }
            )
            frames.append(frame)
            hg_idx += 1

        if verbose:
            print(
                f"Non-uniform config {config_idx + 1}/{len(configs)} complete "
                f"({replicates} replicates)"
            )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _edge_label(number_of_communities: int) -> str:
    return "inter" if number_of_communities > 1 else "intra"


def generate_uniform_data(output_path: Path = UNIFORM_OUTPUT) -> Path:
    """Generate and save the uniform dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_uniform_dataset().to_csv(output_path, index=False)
    return output_path


def generate_nonuniform_data(output_path: Path = NONUNIFORM_OUTPUT) -> Path:
    """Generate and save the 100-replicate non-uniform dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_nonuniform_dataset(verbose=True).to_csv(output_path, index=False)
    return output_path


def main() -> tuple[Path, Path]:
    uniform_path = generate_uniform_data()
    print(f"Uniform HSBM dataset written to {uniform_path}")

    nonuniform_path = generate_nonuniform_data()
    print(f"Non-uniform HSBM dataset written to {nonuniform_path}")
    return uniform_path, nonuniform_path


if __name__ == "__main__":
    main()
