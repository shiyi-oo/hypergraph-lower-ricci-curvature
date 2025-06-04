from itertools import combinations

def generate_complete_hypergraph(n, k):
    hyperedges = list(combinations(range(n),k))
    return hyperedges

def generate_hypercycle(k, s, m):
    """
    Create a hypercycle with m hyperedges, where each hyperedge has k nodes
    and consecutive hyperedges overlap by s nodes.

    Parameters:
    k (int): Number of nodes in each hyperedge.
    s (int): Overlap size between consecutive hyperedges.
    m (int): Number of hyperedges in the hypercycle.

    Returns:
    list: A list of hyperedges, where each hyperedge is a set of nodes.
    """
    if s >= k:
        raise ValueError("Overlap size 's' must be less than the number of nodes 'k' in each hyperedge.")

    hypercycle = []
    # total_nodes = (k - s) * (m+1) - 2*(2*s-k) # Total number of unique nodes needed for the hypercycle
    total_nodes = (k - s) * m

    if total_nodes < k:
        raise ValueError(f"too small m or too large k, where k= {k}, s = {s}, m = {m}")
    nodes = list(range(total_nodes))  # Create a list of nodes

    for i in range(m):
        # Start index for the hyperedge
        start = i * (k - s)
        # Create the hyperedge
        if len(list(nodes[start:start + k]))<k:
            d = k-len(list(nodes[start:start + k]))
            hyperedge = list(nodes[start:start + k]) + (list(nodes[:d]))
        else: 
            hyperedge = list(nodes[start:start + k])
        hypercycle.append(hyperedge)

    return hypercycle

def generate_hypertree(k, r, depth):
    '''
    - k: number of nodes in each hyperedge
    - r: number of edges each node connect
    - d: depth of the hypertree
    
    '''
    
    hyperedge = []
    branch = r-1
    s = 1 # 1-intersecting
    root = [list(range(k))]
    n = k
    for d in range(depth):
        hyperedge += root
        children = []
        for parent in root:
            if n ==k:
                for node in parent[0:]:
                    intersect = [node]
                    for b in range(branch):
                        child = list(range(n,n+k-s))
                        n += k-s
                        children.append(intersect + child)
            else:
                for node in parent[1:]:
                    intersect = [node]
                    for b in range(branch):
                        child = list(range(n,n+k-s))
                        n += k-s
                        children.append(intersect + child)
        root = children
    return hyperedge

