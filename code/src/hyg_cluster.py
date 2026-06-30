import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

def _binning(C, bins):
    def H(c):
        h = np.histogram(c, bins)[0]
        return h / np.sum(h)

    return np.vstack([H(c) for c in C])

def horc_histogram(C):
    return _binning(C, [x / 100 for x in range(-200, 104, 5)])

def cihi_histogram(C):
    return _binning(C, [x / 100 for x in range(-100, 104, 5)])

def kpca(histogram, k):
    pca = KernelPCA(n_components=k, kernel="precomputed", tol=1e-5, max_iter=2000)
    D = rbf_kernel(histogram)
    embedding = pca.fit_transform(D)
    
    return embedding