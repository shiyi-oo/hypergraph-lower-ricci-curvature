"""Bakry--Émery curvature utilities for unweighted graphs.
written by Ben Snodgrass.

This code is based off formulae given in 'Bakry-Émery curvature on graphs as
an eigenvalue problem' by Cushing et al. (2021).

"""

import copy

import numpy as np
from numpy.linalg import eigvalsh


inf = float("inf")


def non_normalised_unweighted_curvature(A, n=inf):
    """Return unnormalised Bakry--Émery curvature for each graph vertex.

    Parameters
    ----------
    A : array-like
        Adjacency matrix of a simple, unweighted graph.
    n : float, default ``inf``
        Curvature dimension.

    Returns
    -------
    list[float]
        Curvature values aligned with the row/column order of ``A``.
    """
    q = len(A)
    curvlst = []

    # Switch to NumPy to increase calculation speed. In this case A[i, j] is
    # p_ij because mu[x] = 1 for all x and there is no edge weighting.
    A = np.array(A, dtype=float)

    # List of one-spheres of the vertices.
    onesps = [[] for _ in range(q)]
    for i in range(q):
        for j in range(q):
            if A[i, j] == 1:
                onesps[i].append(j)

    # Number of nearest neighbours of each vertex, also the degree here.
    lenonesps = [len(onesp) for onesp in onesps]

    # twosp[i, j] = 1 iff i is in the two-sphere of j, otherwise 0.
    A_2 = np.matmul(A, A)
    twosp = copy.copy(A_2)
    for i in range(q):
        for j in range(q):
            if i == j:
                twosp[i, j] = 0
            if A[i, j] == 1:
                twosp[i, j] = 0
            if twosp[i, j] != 0:
                twosp[i, j] = 1

    # Summation terms from Cushing et al. (2021), Appendix A.
    sum2 = np.matmul(A, twosp)
    recipA_2 = twosp * A_2
    for i in range(q):
        for j in range(q):
            if recipA_2[i, j] != 0:
                recipA_2[i, j] = 1 / recipA_2[i, j]

    sum3 = np.matmul(recipA_2, A)
    sum4 = np.matmul(
        [
            [[A[i, z] * A[j, z] for z in range(q)] for j in range(q)]
            for i in range(q)
        ],
        recipA_2,
    )

    for x in range(q):
        m = lenonesps[x]
        if m == 0:
            curvlst.append(0)
            continue

        onesp = onesps[x]
        A_n = [[-2 / n for _ in range(m)] for _ in range(m)]
        for i in range(m):
            A_n[i][i] += (
                5 / 2
                - 1 / 2 * m
                + 2 * A_2[x, onesp[i]]
                + 3 / 2 * sum2[onesp[i], x]
                - 2 * sum3[x, onesp[i]]
            )

        for i in range(m):
            for j in range(m):
                if i != j:
                    A_n[i][j] += (
                        1
                        - 2 * A[onesp[i], onesp[j]]
                        - 2 * sum4[onesp[i], onesp[j], x]
                    )

        # The smallest eigenvalue of A_n is the curvature.
        curvlst.append(round(eigvalsh(A_n)[0], 3))

    return curvlst
