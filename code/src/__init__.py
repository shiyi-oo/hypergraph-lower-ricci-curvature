"""Hypergraph dataset utilities.

Use from another folder by putting the directory that *contains* ``src`` (i.e.
``code/``) on ``sys.path``, then importing the package:

    import sys
    sys.path.append("..")            # if your cwd is a sibling of src/, e.g. code/MADStat
    from src import HypergraphDataset, load_data

    ds = load_data("contact", data_folder="../data")
    print(ds)
"""

from .hg_class import HypergraphDataset
from .data_loader import load_data
from .hellinger_dist import hellinger_distance
from .be import inf, non_normalised_unweighted_curvature
from .community_detect import (
    hypergraph_clique_modularity_maximization,
    hypergraph_spectral_embedding,
    hypergraph_modularity_maximization,
    hypergraph_spectral_clustering,
)
from .hyg_cluster import kpca, cihi_histogram, horc_histogram
from .hsbm import generate_nonuniform_hsbm, generate_uniform_hsbm
from . import HNN
from .HNN import HNNConfig,fit_hgnn_embeddings
