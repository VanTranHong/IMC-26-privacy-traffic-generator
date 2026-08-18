import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import pearson_corr


class CrossFeatureCorrelation(TimeSeriesMetric):
    """This metric computes the Pearson correlation between two features f1 and f2. This correlation is computed (averaged) over every record in every time series in the (real or synthetic) dataset. We then compare the difference in these correlation values between the real and synthetic datasets."""

    name = "Cross Feature Correlation"
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None, target=None, configs=None):
        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)

        if not all(isinstance(s, str) for s in target):
            raise ValueError(
                "target has to be a list of strings where each string specifies an attribute column.")
        assert len(target) == 2, \
            "`target` is expected to be a list including two elements representing two columns."

        column_1 = target[0]
        column_2 = target[1]
        assert metadata['fields'][column_1]['type'] in ['numerical'], \
            f"column {column_1} should be numerical"
        assert metadata['fields'][column_2]['type'] in ['numerical'], \
            f"column {column_2} should be numerical"
        attribute_cols, feature_cols = cls._get_attribute_feature_cols(metadata)
        for col in target:
            if col not in feature_cols:
                raise ValueError(f"Column {col} is not a feature.")

        real_corr = pearson_corr(real_data[column_1].to_numpy(
        ), real_data[column_2].to_numpy())
        synthetic_corr = pearson_corr(
            synthetic_data[column_1].to_numpy(),
            synthetic_data[column_2].to_numpy())

        return [1 - (abs(real_corr - synthetic_corr)) / 2]
