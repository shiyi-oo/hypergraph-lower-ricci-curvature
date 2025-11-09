from .util import *

def compute_hlrc(H: list[list[int]]):
    """
    Compute HLRC curvature for each hyperedge in H (by index) and
    return a list of values (or None if undefined).
    """
    v_neigh = nodes_neighbors(H)
    
    hlrc = []
    for edge in H:
        d_e = len(set(edge))
        if d_e <= 1:
            hlrc.append(None)
            continue
        
        neigh_sizes = [len(v_neigh[v]) for v in edge]
        max_size, min_size = max(neigh_sizes), min(neigh_sizes)
        if max_size == 0 or min_size == 0:
            hlrc.append(None)
            continue
        
        common = set.intersection(*(v_neigh[v] for v in edge))
        n_e = len(common)
        sum_recip = sum(1 / s for s in neigh_sizes)
        
        e_hlrc = (
            sum_recip - 1
            + (n_e + d_e/2 - 1) / max_size
            + (n_e + d_e/2 - 1) / min_size
        )
        hlrc.append(e_hlrc)
    
    return hlrc


def compute_hlrc_naive(H: list[list[int]]):
    """
    Compute HLRC curvature for each hyperedge in H (by index) and
    return a list of values (or None if undefined).
    """
    v_neigh = nodes_neighbors(H)
    
    hlrc = []
    for edge in H:
        d_e = len(set(edge))
        if d_e <= 1:
            hlrc.append(None)
            continue
        
        neigh_sizes = [len(v_neigh[v]) for v in edge]
        max_size, min_size = max(neigh_sizes), min(neigh_sizes)
        if max_size == 0 or min_size == 0:
            hlrc.append(None)
            continue
        
        common = set.intersection(*(v_neigh[v] for v in edge))
        n_e = len(common)
        sum_recip = sum(1 / s for s in neigh_sizes)
        
        e_hlrc = (
            2 * sum_recip - 2
            + (n_e) / max_size
            + (2*n_e) / min_size
        )
        hlrc.append(e_hlrc)
    
    return hlrc