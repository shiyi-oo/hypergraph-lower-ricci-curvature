from util import nodes_degree

def compute_hfrc(H: list[list[int]]) -> list[float]:
    """
    Compute H-FRC curvature for each hyperedge.
    """
    Dv = nodes_degree(H)
    hfrc: list[float] = []
    for edge in H:
        d_e = len(edge)
        deg_sum = sum(Dv[v] for v in edge)
        hfrc.append(2 * d_e - deg_sum)
    return hfrc
