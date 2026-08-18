import warnings
import numpy as np
import torch
from geomloss import SamplesLoss
from typing import Optional, Dict, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from collections import Counter, OrderedDict
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from .misc import get_frequencies


def jsd(p: np.ndarray, q: np.ndarray, categorical_mapping: bool):
    """Compute the Jensen-Shannon distance (metric) between two distributions. This is the square root of the Jensen-Shannon divergence.

    Args: 
        p: nxd array.
        q: mxd array.
            p and q can have different number of samples (n \neq m), but the dimensionality has to be the same (d).
        categorical_mapping: exact mapping for categorical variable or not.
    """
    assert len(p.shape) == len(q.shape) == 2, \
        "p and q must be arrays of shape (n, d). " \
        "n is the number of samples, d is the dimensionality of each sample."

    assert p.shape[1] == q.shape[1], \
        "p and q must have the same dimensionality."

    # 1D
    if p.shape[1] == 1:
        f_p, f_q = get_frequencies(p.T[0], q.T[0], categorical_mapping)
        return distance.jensenshannon(f_p, f_q)
    # multi(>=2)-dimension
    elif p.shape[1] > 1:
        f_p, f_q = get_frequencies(
            list(map(tuple, p)), list(map(tuple, q)), categorical_mapping)
        return distance.jensenshannon(f_p, f_q)
    else:
        raise ValueError("Invalid dimensions!")


def emd(p: np.ndarray, q: np.ndarray):
    """Compute the Wasserstein distance between two distributions.

    For 1D arrays, use scipy.stats.wasserstein_distance().
    For higher dimensional (>=2) arrays, use GeomLoss library for approximation.

    Args:
        p: nxd array.
        q: mxd array.
            p and q can have different number of samples (n \neq m), but the dimensionality has to be the same (d).
    """
    assert len(p.shape) == len(q.shape) == 2, \
        "p and q must be arrays of shape (n, d). " \
        "n is the number of samples, d is the dimensionality of each sample."

    assert p.shape[1] == q.shape[1], \
        "p and q must have the same dimensionality."

    # d=1
    if p.shape[1] == 1:
        return wasserstein_distance(p.T[0], q.T[0])
    # d>=2
    elif p.shape[1] > 1:
        use_cuda = torch.cuda.is_available()
        Loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
        wass_pq = Loss(torch.Tensor(p), torch.Tensor(q))
        if use_cuda:
            torch.cuda.synchronize()
        return wass_pq.item()
