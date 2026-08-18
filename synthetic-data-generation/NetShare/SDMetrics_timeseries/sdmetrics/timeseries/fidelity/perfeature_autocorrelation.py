import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import autocorrelation_similarity


class PerFeatureAutocorrelation(TimeSeriesMetric):
    """Checks the autocorrelation for feature f’s time series (compared to the real data)"""

    name = "PerFeatureAutocorrelation"
    goal = Goal.MINIMIZE
    min_value = 0.0
    max_value = float("inf")

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None, target=None, configs=None):
        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)

        if not all(isinstance(s, str) for s in target):
            raise ValueError(
                "target has to be a list of strings where each string specifies an attribute column.")
        assert len(target) == 1, \
            "`target` is expected to be a single-element list where the only element in the feature column to be computed."

        attribute_cols, feature_cols = cls._get_attribute_feature_cols(metadata)
        for col in target:
            if col not in feature_cols:
                raise ValueError(f"Column {col} is not a feature.")
            if metadata['fields'][col]['type'] != 'numerical':
                raise ValueError(f"Column {col} is not a numerical feature")

        real_gk = real_data.groupby(attribute_cols)
        real_feature = []
        synthetic_gk = synthetic_data.groupby(attribute_cols)
        synthetic_feature = []
        max_length = max(max(real_gk.size()), max(synthetic_gk.size()))

        for group_name, df_group in real_gk:
            real_feature.append(list(df_group[target[0]]) +
                                [0.0] * (max_length - len(df_group)))
        real_feature = np.asarray(
            real_feature).reshape(-1, max_length)
        for group_name, df_group in synthetic_gk:
            synthetic_feature.append(list(df_group[target[0]]) +
                                     [0.0] * (max_length - len(df_group)))
        synthetic_feature = np.asarray(
            synthetic_feature).reshape(-1, max_length)

        return autocorrelation_similarity(
            real_data=real_feature,
            synthetic_data=synthetic_feature,
            column_names=target,
            data_type=[metadata['fields'][col]['type'] for col in target],
            comparison_type='both'
        )
