import numpy as np
from math import sqrt
def hellinger_distance(samples_a, samples_b, bins=200, value_range=None):
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    combined = np.concatenate([a, b])
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return np.nan

    if value_range is None:
        data_min = combined.min()
        data_max = combined.max()
    else:
        data_min, data_max = value_range

    if not np.isfinite(data_min) or not np.isfinite(data_max):
        return np.nan
    if data_min == data_max:
        return 0.0

    edges = np.linspace(data_min, data_max, bins + 1)
    counts_a, _ = np.histogram(a, bins=edges)
    counts_b, _ = np.histogram(b, bins=edges)

    widths = np.diff(edges)
    total_a = counts_a.sum()
    total_b = counts_b.sum()
    if total_a == 0 or total_b == 0 or not np.any(widths):
        return np.nan

    pdf_a = counts_a / total_a / widths
    pdf_b = counts_b / total_b / widths
    diff = np.sqrt(pdf_a) - np.sqrt(pdf_b)
    distance = (1.0 / sqrt(2.0)) * np.sqrt(np.sum(diff ** 2 * widths))
    if not np.isfinite(distance):
        return np.nan
    return float(np.clip(distance, 0.0, 1.0))