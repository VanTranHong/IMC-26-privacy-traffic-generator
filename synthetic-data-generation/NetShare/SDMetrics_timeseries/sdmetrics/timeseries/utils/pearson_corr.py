import numpy as np
from scipy.stats import pearsonr


def pearson_corr(p: np.ndarray, q: np.ndarray):
    """Compute the pearson correlation between two distributions.

    Only for 1D arrays, use scipy.stats.pearsonr().

    Args:
        p: nxd array.
        q: mxd array.
    """

    assert p.shape[0] == q.shape[0], \
        "p and q must have the same dimensionality which should be 1."
    res = pearsonr(p, q)
    return res.statistic
