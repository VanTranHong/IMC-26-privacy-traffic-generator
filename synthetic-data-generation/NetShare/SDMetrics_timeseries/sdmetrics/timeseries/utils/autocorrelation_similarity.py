import warnings
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
from typing import Optional, Dict, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from collections import Counter, OrderedDict
from .distance import jsd, emd
from sdmetrics.reports.utils import make_discrete_column_plot, make_continuous_column_plot

EPS = 1e-8


def autocorr(X, Y):
    Xm = torch.mean(X, 1).unsqueeze(1)
    Ym = torch.mean(Y, 1).unsqueeze(1)
    r_num = torch.sum((X - Xm) * (Y - Ym), 1)
    r_den = torch.sqrt(torch.sum((X - Xm)**2, 1) * torch.sum((Y - Ym)**2, 1))

    r_num[r_num == 0] = EPS
    r_den[r_den == 0] = EPS

    r = r_num / r_den
    r[r > 1] = 0
    r[r < -1] = 0

    return r


def get_autocorr(feature):
    feature = torch.from_numpy(feature)
    feature_length = feature.shape[1]
    autocorr_vec = torch.Tensor(feature_length - 2)

    for j in range(1, feature_length - 1):
        autocorr_vec[j - 1] = torch.mean(autocorr(feature[:, :-j],
                                                  feature[:, j:]))

    return autocorr_vec


def autocorrelation_similarity(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    column_names: List[str],
    data_type: List[Literal['categorical', 'numerical']],
    comparison_type: Literal['quantitative', 'qualitative', 'both'],
):
    assert len(column_names) == 1, \
        "The length of columns should be 1."

    assert len(data_type) == 1, \
        "Only one column of data is accepted."

    assert data_type[0] == "numerical", \
        "Only numerical column is accepted."

    output = []

    real_data = real_data.astype(float)
    synthetic_data = synthetic_data.astype(float)

    real_autocorr = get_autocorr(real_data).numpy()
    synthetic_autocorr = get_autocorr(synthetic_data).numpy()

    if comparison_type in ['quantitative', 'both']:
        output.append(
            np.sum(np.abs(synthetic_autocorr - real_autocorr)) / np.sum(real_autocorr))

    if comparison_type in ['qualitative', 'both']:
        output.append(make_continuous_column_plot(
            real_column=pd.Series(
                real_data.flatten(), name=column_names[0]),
            synthetic_column=pd.Series(
                synthetic_data.flatten(), name=column_names[0]),
            sdtype='numerical'
        ))
    return output
