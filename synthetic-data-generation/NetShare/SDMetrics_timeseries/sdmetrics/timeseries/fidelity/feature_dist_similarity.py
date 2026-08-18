import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import distribution_similarity


class FeatureDistSimilarity(TimeSeriesMetric):
    """We compute the marginal distribution of values for a single fixed feature f. For example, for the feature key “latitude”, this would compute a distribution over all the latitudes seen in every time series in the dataset. We compare this distribution between synthetic and real data.

    Fix a set of features F={f1,...,fm} whose joint distribution you want to compare. (This can be used to evaluate joint distributions or marginal distributions. If you only want to look at marginals (ie one feature at a time) just let F contain a single feature)."""

    name = "Feature distributional similarity"
    goal = Goal.MINIMIZE

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None, target=None, configs=None):
        if not all(isinstance(s, str) for s in target):
            raise ValueError(
                "target has to be a list of strings where each string specifies a feature column.")

        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)
        attribute_cols, feature_cols = cls._get_attribute_feature_cols(metadata)
        for col in target:
            if col not in feature_cols:
                raise ValueError(f"Column {col} is not a feature.")

            # Convert datetime to unix timestamp (unit: second) in-place
            if metadata['fields'][col]['type'] == 'datetime':
                real_data[col] = pd.to_datetime(
                    real_data[col]).astype(int) / 10**9
                synthetic_data[col] = pd.to_datetime(
                    synthetic_data[col]).astype(int) / 10**9

        if metadata['fields'][target[0]]['type'] in ['numerical', 'datetime']:
            cls.min_value = 0.0
            cls.max_value = float("inf")
        elif metadata['fields'][target[0]]['type'] in ['categorical']:
            cls.min_value = 0.0
            cls.max_value = 1.0

        real_columns = real_data[target].to_numpy().reshape(-1, len(target))
        synthetic_columns = synthetic_data[target].to_numpy(
        ).reshape(-1, len(target))

        return distribution_similarity(
            real_data=real_columns,
            synthetic_data=synthetic_columns,
            column_names=target,
            data_type=['numerical'
                       if metadata['fields'][col]['type'] == 'datetime' else
                       metadata['fields'][col]['type'] for col in target],
            comparison_type=getattr(configs, 'comparison_type', 'both'),
            categorical_mapping=getattr(configs, 'categorical_mapping', True))
