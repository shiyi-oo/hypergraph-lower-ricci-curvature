import numpy as np
import pandas as pd
from itertools import combinations

import scipy.sparse.linalg as la

import matplotlib.pyplot as plt
import xgi
import matplotlib

import seaborn as sns

import numpy as np
from itertools import combinations

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'savefig.bbox': 'tight',
    'savefig.transparent':True})

def generate_hsbm(
    k_uniform: int,
    block_sizes: list[int],
    community_edge_probs: list[float],
    seed: int = 123
) -> tuple[list[list[int]], list[int], list[int]]:
    """
    Builds a k-uniform Hypergraph Stochastic Block Model.
    
    Parameters:
    -----------
    k_uniform : int
        Size of each hyperedge.
    block_sizes : list[int]
        Number of nodes in each community. Sum(block_sizes) = total nodes.
    community_edge_probs : list[float]
        Probabilities for including an edge based on how many distinct communities
        its nodes span. Index 0 = all nodes from one community, index 1 = two communities, etc.
    seed : int (default=123)
        Random seed for reproducibility.
    
    Returns:
    --------
    hyperedges : list[set[int]]
        Each retained hyperedge as a list of node indices.
    node_labels : list[int]
        For each node index, the community label it belongs to.
    edge_community_counts : list[int]
        Number of distinct communities spanned by each retained hyperedge.
    """
    # Initialize RNG
    np.random.seed(seed)
    
    # Total nodes and flat list of node indices
    total_nodes = sum(block_sizes)
    all_nodes = list(range(total_nodes))
    
    # Assign each node to a community label
    node_labels: list[int] = []
    for community_idx, size in enumerate(block_sizes):
        node_labels.extend([community_idx] * size)
    
    # Enumerate all possible k-node combinations
    candidate_edges = combinations(all_nodes, k_uniform)
    
    # Build list of hyperedges
    hyperedges: list[set[int]] = []
    edge_community_counts: list[int] = []
    
    for edge in candidate_edges:
        # Which communities are represented in this edge?
        distinct_comms = {node_labels[node] for node in edge}
        num_comms = len(distinct_comms)
        
        # Pick the inclusion probability based on number of communities
        prob = community_edge_probs[num_comms - 1]
        
        # Randomly decide to keep this edge
        if np.random.rand() < prob:
            hyperedges.append(set(edge))
            edge_community_counts.append(num_comms)
    
    return hyperedges, node_labels, edge_community_counts



def plot_edge_hlrc_boxplot(df, save=False,title=None ):
    plt.subplots(figsize=(2, 3))
    spec = dict(data = df, x = "edge_label", y = "hlrc", hue="edge_label", width = .6)
    sns.boxplot(**spec, linewidth=0, showfliers=False, boxprops=dict(alpha=.5), palette={"intra": "#c46666","inter": "#1a80bb"})
    sns.boxplot(**spec, linewidth=1, fill=False, legend=False, palette={"intra": "#c46666","inter": "#1a80bb"} )
    plt.yticks([-1.0,-0.5,0.0, 0.5, 1.0]);
    plt.xlabel("")
    plt.ylabel("HLRC")
    if title != None:
        plt.title(title)
    if save:
        filepath = f"./figures/boxplot_{title}.pdf"
        plt.savefig(filepath, bbox_inches='tight')

def plot_graph(H: list[list], hlrc: list[list], node_label,
               save = False, title=None, seed=123, node_size=4, layout = None):
    H = xgi.Hypergraph(H)
    # node position
    if layout == "barycenter":
        pos = xgi.barycenter_spring_layout(H)
    else: pos = xgi.pairwise_spring_layout(H, seed=seed, iterations=100)
    # edge color
    cmap = matplotlib.colormaps.get_cmap('RdYlBu_r')
    edge_color = {key: cmap((cuvr+1)/2) for key, cuvr in enumerate(hlrc)}

    # plot
    fig, ax = plt.subplots(figsize=(5, 5))
    xgi.draw(
        H, 
        pos=pos,
        node_size=node_size,
        node_lw = 0,
        node_ec = "black",
        node_fc = 'black', # node color
        edge_fc = edge_color, # edge color
        edge_ec = "white",
        alpha=0.4,
        hull = True,
        radius = 0.03,
        rescale_sizes=False);
    if title != None:
        plt.title(title)
    if save:
        savepath=f'./figures/graph_{title}.pdf'
        plt.savefig(savepath,bbox_inches='tight');
    
    plt.show();