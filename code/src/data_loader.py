import argparse
import gzip
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET = ["contact", "madstat", "mag10", "mus", "stex"]

try:
    from .hg_class import HypergraphDataset      # imported as part of the `src` package
except ImportError:
    from hg_class import HypergraphDataset        # run directly as a script from inside src/



# --------------------------------------------------------------------------- #
# Raw loaders: parse the original files under data/raw/{dname}/                #
# --------------------------------------------------------------------------- #
def load_contact(raw_path="../data/raw/contact"):
    with open(raw_path + "/hyperedges-contact-high-school.txt", "r") as f:
        hyperedges = [list(map(int, line.strip().split(","))) for line in f]

    with open(raw_path + "/label-names-contact-high-school.txt", "r") as f:
        label_names = [line.strip() for line in f]

    with open(raw_path + "/node-labels-contact-high-school.txt", "r") as f:
        node_label_idx = np.array([int(line.strip()) for line in f])
    node_label_idx = node_label_idx - node_label_idx.min()  # normalize to 0-based

    node_labels = [label_names[idx] for idx in node_label_idx]
    return HypergraphDataset(hyperedges, node_labels=node_labels)


def load_madstat(raw_path="../data/raw/madstat"):
    madstat = pd.read_csv(raw_path + "/AuPapMat.txt")
    madstat_hg = madstat.groupby("idxPap", as_index=False).agg(
        hyperedges=("idxAu", list),
        edge_year=("year", "first"),
        edge_journal=("journal", "first"),
    )
    hyperedges = madstat_hg["hyperedges"].tolist()
    edge_labels = madstat_hg[["edge_year", "edge_journal"]].values.tolist()
    with open(raw_path + "/author_name.txt", "r") as f:
        author_names = [line.strip() for line in f]
    return HypergraphDataset(hyperedges, node_labels=author_names,
                             edge_labels=edge_labels)


def load_mag10(raw_path="../data/raw/mag10"):
    with open(raw_path + "/hyperedges.txt", "r") as f:
        hyperedges = [list(map(int, line.strip().split("\t"))) for line in f]

    with open(raw_path + "/hyperedge-label-identities.txt", "r") as f:
        label_names = [line.strip() for line in f]  # ['KDD', 'WWW', 'ICML', ...]

    with open(raw_path + "/hyperedge-labels.txt", "r") as f:
        label_idx = np.array([int(line.strip()) for line in f])  # [4, 10, 8, ...]
    label_idx = label_idx - label_idx.min()  # normalize to start from 0

    # keep only the first 10 venue labels
    filtered = [(idx, e) for idx, e in zip(label_idx, hyperedges) if idx < 10]
    label_idx, hyperedges = (list(t) for t in zip(*filtered))
    hg_labels = [label_names[idx] for idx in label_idx]
    return HypergraphDataset(hyperedges, hypergraph_labels=hg_labels)


def load_mus(raw_path="../data/raw/mus"):
    hg_idx = []
    hyperedges = []
    with gzip.open(raw_path + "/mus.chg.tsv.gz", "rt") as f:
        for line in tqdm(f, desc="Processing rows"):
            columns = list(map(int, line.strip().split("\t")))
            hg_idx.append(columns[0])      # hypergraph id
            hyperedges.append(columns[1:])  # member nodes
    meta = pd.read_csv(raw_path + "/hg_labels.txt", sep="\t", header=None,
                       names=["hg_idx", "label"],
                       dtype={"hg_idx": int, "label": str})
    hg_labels = meta["label"].tolist()
    return HypergraphDataset(hyperedges, hypergraph_idx=hg_idx,
                             hypergraph_labels=hg_labels)


def load_stex(raw_path="../data/raw/stex"):
    hg_idx = []
    hyperedges = []
    with gzip.open(raw_path + "/stex.chg.tsv.gz", "rt") as f:
        for line in tqdm(f, desc="Processing rows"):
            columns = list(map(int, line.strip().split("\t")))
            hg_idx.append(columns[0])      # hypergraph id
            hyperedges.append(columns[1:])  # member nodes
    hg_labels = []
    with open(raw_path + "/stex_site_id.txt", "r") as f:
        for line in f:
            _idx, label = line.strip().split("\t")
            hg_labels.append(label)
    return HypergraphDataset(hyperedges, hypergraph_idx=hg_idx,
                             hypergraph_labels=hg_labels)


_RAW_LOADERS = {
    "contact": load_contact,
    "madstat": load_madstat,
    "mag10": load_mag10,
    "mus": load_mus,
    "stex": load_stex,
}


def load_data(dname, raw_path=None, processed_path=None, data_folder=None,
              use_cache=True):
    """Load a hypergraph dataset as a :class:`HypergraphDataset`.

    Cache + auto-fallback:
      1. If a processed pickle exists, the class is unpickled from it.
      2. Otherwise the raw files are parsed and the result is saved to the
         processed pickle for next time.
      3. Lazily computed curvature values are subsequently written back to the
         same pickle by the dataset's ``get_*`` methods.

    Paths may be given explicitly via ``raw_path`` / ``processed_path``, or
    derived from ``data_folder`` (``{data_folder}/raw/{dname}`` and
    ``{data_folder}/processed/{dname}.pkl``).
    """
    if dname not in _RAW_LOADERS:
        raise ValueError(f"Unknown dataset: {dname}")

    if data_folder is not None:
        if raw_path is None:
            raw_path = f"{data_folder}/raw/{dname}"
        if processed_path is None:
            processed_path = f"{data_folder}/processed/{dname}.pkl"

    # 1. load from the processed cache when available
    if use_cache and processed_path is not None and os.path.exists(processed_path):
        return HypergraphDataset.load(processed_path)

    # 2. build from raw
    if raw_path is None:
        raise ValueError(
            "No cached pickle found; raw_path (or data_folder) is required "
            f"to build dataset '{dname}' from raw files.")
    dataset = _RAW_LOADERS[dname](raw_path)

    # 3. populate the cache for next time
    if processed_path is not None:
        dataset.save(processed_path)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Load a hypergraph dataset")
    parser.add_argument("--dname", default="contact", required=False, choices=DATASET,
                        help="Name of the dataset to load")
    parser.add_argument("--data_folder", default="../data",
                        help="Path to the dataset folder")
    parser.add_argument("--no_cache", action="store_true",
                        help="Re-parse raw files even if a processed pickle exists")
    args = parser.parse_args()

    dataset = load_data(args.dname, data_folder=args.data_folder,
                        use_cache=not args.no_cache)

    print(dataset)


if __name__ == "__main__":
    main()
