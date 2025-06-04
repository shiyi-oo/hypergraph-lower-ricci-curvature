import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# def nodes_idx(H: dict):
#     # Use a set comprehension and sort the result
#     return sorted({node for nodes in H.values() for node in nodes})

# H: List of hyperedges, each hyperedge is a List of nodes
def nodes_idx(H: list[list[int]]) -> list[int]:
    """All unique nodes, sorted."""
    return sorted({node for edge in H for node in edge})


# def nodes_degree(H: dict):
#     """
#     Calculate the degree of each node in the hypergraph using Counter for efficiency.
#     """
#     return Counter(node for nodes in H.values() for node in nodes)
def nodes_degree(H: list[list[int]]) -> Counter:
    """
    Degree of each node (i.e. in how many hyperedges it appears).
    """
    return Counter(node for edge in H for node in edge)


# def nodes_neighbors(H: dict):
#     """
#     Compute the neighborhood for each vertex in the hypergraph.
#     """
#     neighbors = defaultdict(set)
#     for nodes in H.values():
#         nodes_set = set(nodes)
#         for v in nodes:
#             # Update with all nodes in the edge except itself
#             neighbors[v].update(nodes_set - {v})
#     return dict(neighbors)

def nodes_neighbors(H: list[list[int]]) -> dict[int, set[int]]:
    """
    For each node, the set of all other nodes with which it co-occurs in any hyperedge.
    """
    v_neigh: dict[int, set[int]] = defaultdict(set)
    for edge in H:
        s = set(edge)
        for v in edge:
            v_neigh[v].update(s - {v})
    return dict(v_neigh)

# def edges_neighborhood(H: dict, neighbors: dict):
#     """
#     Compute the common neighborhood (intersection of neighborhoods) for each hyperedge.
#     """
#     neighborhood = {}
#     for edge, nodes in H.items():
#         if len(nodes) <= 1:
#             neighborhood[edge] = set()
#         else:
#             neighborhood[edge] = set.intersection(*(neighbors[v] for v in nodes))
#     return neighborhood

def edges_neighborhood(H: list[list[int]],
                       v_neigh: dict[int, set[int]]
                      ) -> dict[int, set[int]]:
    """
    For each hyperedge (by its index in H), the intersection of its member-nodes' neighborhoods.
    """
    e_neigh: dict[int, set[int]] = defaultdict(set)
    for idx, edge in enumerate(H):
        if len(edge) <= 1:
            e_neigh[idx] = set()
        else:
            # intersection of neighbors[v] for v in edge
            e_neigh[idx] = set.intersection(*(v_neigh[v] for v in edge))
    return e_neigh


# def compute_neighborhood_sizes(H: dict, neighbors: dict):
#     """
#     Compute for each hyperedge: 
#       (common_neighbors_size, max_neighborhood_size, min_neighborhood_size).
#     """
#     neighborhood_size_dict = {}
#     for edge, nodes in H.items():
#         # Compute neighborhood sizes once
#         sizes = [len(neighbors[v]) for v in nodes]
#         common = set.intersection(*(neighbors[v] for v in nodes)) if nodes else set()
#         neighborhood_size_dict[edge] = (len(common), max(sizes), min(sizes))
#     return neighborhood_size_dict

def compute_neighborhood_sizes(
    H: list[list[int]],
    v_neigh: dict[int, set[int]]
) -> dict[int, tuple[int,int,int]]:
    """
    For each hyperedge idx:
      ( size_of_common_neighbors,
        max_individual_neighborhood_size,
        min_individual_neighborhood_size )
    """
    size_dict: dict[int, tuple[int,int,int]] = defaultdict(set)
    for idx, edge in enumerate(H):
        if not edge:
            size_dict[idx] = (0, 0, 0)
            continue
        # sizes of each node's neighbor‐set
        sizes = [len(v_neigh[v]) for v in edge]
        common = set.intersection(*(v_neigh[v] for v in edge))
        size_dict[idx] = (len(common), max(sizes), min(sizes))
    return size_dict


def get_incidence_matrix(H: list[list[int]]) -> np.ndarray:
    """
    Incidence matrix (nodes × hyperedges) as a NumPy array:
      rows = all nodes, sorted
      columns = hyperedge indices (0,1,2,…)
      entry is 1 if node ∈ edge, else 0
    """
    all_nodes = nodes_idx(H)  # your existing function returning sorted list of nodes
    node_index = {n: i for i, n in enumerate(all_nodes)}
    M = np.zeros((len(all_nodes), len(H)), dtype=int)

    for e_idx, edge in enumerate(H):
        for n in edge:
            M[node_index[n], e_idx] = 1

    return M

def get_summary_stat(H: list[list[int]]):
    inc = get_incidence_matrix(H)
    num_nodes, num_edges = inc.shape
    print("# of edges:", num_edges)
    print("# of nodes:", num_nodes)
    
    node_deg = inc.sum(axis=1)
    mean_node_deg = node_deg.mean()
    std_node_deg  = node_deg.std(ddof=0)
    print(f"node degree: - average: {mean_node_deg:.1f}; sd:{std_node_deg:.1f}")
    edge_deg = inc.sum(axis=0)
    mean_edge_deg = edge_deg.mean()
    std_edge_deg  = edge_deg.std(ddof=0)
    print(f"edge degree: - average {mean_edge_deg:.1f}; sd: {std_edge_deg:.1f}")
    
    max_edge_deg = int(edge_deg.max()) if num_edges > 0 else 0
    print("max edge degree:", max_edge_deg)
    
    num_deg2 = int((edge_deg == 2).sum())
    pct_deg2 = (num_deg2 / num_edges) * 100 if num_edges > 0 else 0.0
    print(f"percent of edges with degree 2: {pct_deg2:.1f}%")