import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import distribution_similarity


class SessionLengthDistSimilarity(TimeSeriesMetric):
    name = "Session length distributional similarity"
    goal = Goal.MINIMIZE
    min_value = 0.0
    max_value = float("inf")

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None, configs=None):
        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)
        attribute_cols, feature_cols = cls._get_attribute_feature_cols(metadata)

        column_name = "session_length"
        real_sess_length = real_data.groupby(
            attribute_cols).size().reset_index(name=column_name)
        synthetic_sess_length = synthetic_data.groupby(
            attribute_cols).size().reset_index(name=column_name)
        real_column = real_sess_length[column_name].to_numpy().reshape(-1, 1)
        synthetic_column = synthetic_sess_length[column_name].to_numpy(
        ).reshape(-1, 1)

        return distribution_similarity(
            real_data=real_column,
            synthetic_data=synthetic_column,
            column_names=[column_name],
            data_type=['numerical'],
            comparison_type=getattr(configs, 'comparison_type', 'both'),
            categorical_mapping=True
        )
