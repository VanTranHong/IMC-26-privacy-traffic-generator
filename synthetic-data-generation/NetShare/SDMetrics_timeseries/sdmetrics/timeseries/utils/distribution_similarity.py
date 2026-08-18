import warnings
import numpy as np
import pandas as pd
import torch
from geomloss import SamplesLoss
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import Optional, Dict, List
from collections import Counter, OrderedDict
from .distance import jsd, emd
from sdmetrics.reports.utils import make_discrete_column_plot, make_continuous_column_plot


# TODO: Check emd error if real_data and synthetic_data is identical.
def distribution_similarity(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    column_names: List[str],
    data_type: List[Literal['categorical', 'numerical']],
    comparison_type: Literal['quantitative', 'qualitative', 'both'],
    categorical_mapping: Optional[bool] = None
):
    """Computes the quantitative and/or qualitative similarity between two distributions

    Inputs:
        Synthetic data (represented as a data array)
        Real data (represented as a data array)
        Column names (name of each dimension in real/synthetic data)
        Data type (categorical, continuous)
        Type of comparison (quantitative, qualitative, both)
        Categorical mapping: search exact mapping of values
    """
    assert len(real_data.shape) == len(synthetic_data.shape) == 2, \
        "Both real data and synthetic data must be 2D array. " \
        "For 1D array, use reshape(-1, 1) for conversion."

    assert real_data.shape[1] == synthetic_data.shape[1] \
        == len(column_names) == len(data_type), \
        "Real data and synthetic data must have the same dimension. " \
        "Each dimension of data has to be assigned a name. " \
        "Each dimension of data has to be speicified as `categorical` or `numerical`."

    output = []

    # Quantitative
    if comparison_type in ['quantitative', 'both']:
        # categorical only
        if set(data_type) == {'categorical'}:
            assert categorical_mapping is not None, \
                "Categorical variable, `categorical_mapping` must be set."
            output.append(jsd(real_data, synthetic_data, categorical_mapping))

        # numerical only
        elif set(data_type) == {'numerical'}:
            output.append(emd(real_data, synthetic_data))

        # categorical/numerical mixed: discretizing numerical variables
        # TODO: think of alternative strategy
        elif set(data_type) == {'categorical', 'numerical'}:
            raise warnings.warn(
                "Multiple varibles with categorical and numerical values mixed! Currently not supported for quantitative comprasion.")
        else:
            raise ValueError(
                "Unsupported data type, only `categorical` and `numerical` are supported.")

    # Qualitative
    if comparison_type in ['qualitative', 'both']:
        # 1D array
        if real_data.shape[1] == 1:
            if data_type[0] == 'categorical':
                output.append(make_discrete_column_plot(
                    real_column=pd.Series(
                        real_data.flatten(), name=column_names[0]),
                    synthetic_column=pd.Series(
                        synthetic_data.flatten(), name=column_names[0]),
                    sdtype='categorical'))
            elif data_type[0] == 'numerical':
                output.append(make_continuous_column_plot(
                    real_column=pd.Series(
                        real_data.flatten(), name=column_names[0]),
                    synthetic_column=pd.Series(
                        synthetic_data.flatten(), name=column_names[0]),
                    sdtype='numerical'
                ))
        # 2D array
        # TODO:

    return output
