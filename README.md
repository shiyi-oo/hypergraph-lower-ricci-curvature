# Curvature-Induced Hypergraph Index

This git repo includes both data and codes to reproduce results given in paper "Lower Ricci Curvature for Hypergraphs".

Yang, S., Chen, C., & Li, D. (2025). Lower Ricci Curvature for Hypergraphs. arXiv preprint arXiv:2506.03943. 

Paper link: [https://arxiv.org/pdf/2506.03943](https://arxiv.org/pdf/2506.03943)

## Repo Structure
Code: 
  - src: includes the source code to caculate HLRC and HFRC, while the computation of HORC in julia should refers to the implementation given in https://github.com/aidos-lab/orchid.git.
  - special_hypg, hsbm: special uniform hypergraphs were visualized, and synthetic hypergraphs based on stochastic block model were generated and evaluated.
  - contact-high school, MADStat, MAG-10, Mus, Stex: the real-world dataset evaluation
      1. Data process
      2. Curvature computation
      3. Visualization/Comparions
  - runtime: generated synthetic Chung–Lu hypergraphs as the number of hyperedges, the number of nodes, and the average hyperedge size varied one at a time.

Data: all raw data.
