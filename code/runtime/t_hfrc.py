import time
import re

import sys
import os
sys.path.append(os.path.abspath('/work/users/s/h/shiyi/hypergraph_with_curvature/code/src'))
from hfrc import compute_hfrc

def parse_syncl_name(file_name: str):
    """
    Parse filenames like "syn_cl(n,m,k=100,1000,4).tsv"
    and return the integers (n, m, k).
    """
    # regex with three capture groups for the digits
    pattern = r"^syn_cl\(n,m,k=(\d+),(\d+),(\d+)\)\.tsv$"
    base = os.path.basename(file_name)
    m = re.match(pattern, base)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {file_name!r}")
    # Convert captured strings to ints
    n_val, m_val, k_val = map(int, m.groups())
    return n_val, m_val, k_val

def load_tsv(path: str) -> list[list[int]]:
    hyperedges: list[list[int]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            hyperedge = list(map(int, line.strip().split("\t")))
            hyperedges.append(hyperedge)
    return hyperedges

file_dir = './derived_data/'
results = []
for file_name in os.listdir(file_dir):
    if file_name.endswith('.tsv'):

        n,m,k=parse_syncl_name(file_name)
        file_path = os.path.join(file_dir, file_name)
        
        hyperedges = load_tsv(file_path)

        # Record computation time for compute_hfrc(H_dict)
        start_time = time.time()
        compute_hfrc(hyperedges)
        t_hfrc = time.time() - start_time

        # Store the results
        results.append((n, m, k, t_hfrc))

# Save the results into a txt file
results_file = "output/t_hfrc.txt"
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, 'w') as f:
    # Write the header
    f.write("n\tm\tk\tt_hfrc\n")
    
    # Write each row of results
    for result in results:
        f.write(f"{result[0]}\t{result[1]}\t{result[2]}\t{result[3]:.2f}\n")

print(f"Computation times saved to {results_file}")
