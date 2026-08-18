import numpy as np
import pandas as pd
from typing import Optional, Dict, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from collections import Counter, OrderedDict
from sdmetrics.reports.utils import make_venn2_plot, make_overlap_range_1d_plot


def jaccard_similarity(A: set, B: set):
    # Find intersection of two sets
    nominator = A.intersection(B)

    # Find union of two sets
    denominator = A.union(B)

    # Take the ratio of sizes
    similarity = len(nominator)/len(denominator)

    return similarity


def coverage(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    column_names: List[str],
    data_type: List[Literal['categorical', 'numerical']],
    comparison_type: Literal['quantitative', 'qualitative', 'both']
):
    """Compares the support of the real and synthetic data (taken from the metadata) using Jaccard similarity. 

    Inputs:
        Synthetic data (represented as a data array)
        Real data (represented as a data array)
        Column names (name of each dimension in real/synthetic data)
        Data type (categorical, continuous)
        Type of comparison (quantitative, qualitative, both)
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
            if real_data.shape[1] == 1:
                real_support, synthetic_support = \
                    set(list(real_data.flatten())), \
                    set(list(synthetic_data.flatten()))
            elif real_data.shape[1] > 1:
                real_support, synthetic_support = \
                    set(list(map(tuple, real_data))),
                set(list(map(tuple, synthetic_data)))
            else:
                raise ValueError("Invalid dimensions!")
            output.append(jaccard_similarity(real_support, synthetic_support))

        # numerical only
        elif set(data_type) == {'numerical'}:
            if real_data.shape[1] == 1:
                real_data_flatten, synthetic_data_flatten = \
                    real_data.flatten(), synthetic_data.flatten()
                numerator = (min(
                    max(real_data_flatten),
                    max(synthetic_data_flatten)) -
                    max(
                    min(real_data_flatten),
                    min(synthetic_data_flatten)))
                denominator = (max(
                    max(real_data_flatten),
                    max(synthetic_data_flatten)) -
                    min(
                    min(real_data_flatten),
                    min(synthetic_data_flatten)))
                # Coverage is zero if real and synthetic data does not overlap
                output.append(numerator / denominator
                              if numerator >= 0 else 0.0)
            # TODO: efficient implementation
            elif real_data.shape[1] > 1:
                pass
            else:
                raise ValueError("Invalid dimensions!")

        # TODO: mixed categorical/numerical: Not valid
        elif set(data_type) == {'categorical', 'numerical'}:
            raise ValueError(
                "`Coverage` metric is not applicable for mixed types of variables.")
        else:
            raise ValueError(
                "Unsupported data type, only `categorical` and `numerical` are supported.")

    # Qualitative
    if comparison_type in ['qualitative', 'both']:
        # 1D array
        if real_data.shape[1] == 1:
            if data_type[0] == 'categorical':
                output.append(
                    make_venn2_plot(
                        real_support=real_support,
                        synthetic_support=synthetic_support,
                        column_name=column_names[0]))
            elif data_type[0] == 'numerical':
                output.append(
                    make_overlap_range_1d_plot(
                        real_data=real_data.flatten(),
                        synthetic_data=synthetic_data.flatten(),
                        column_name=column_names[0]
                    )
                )

    return output
