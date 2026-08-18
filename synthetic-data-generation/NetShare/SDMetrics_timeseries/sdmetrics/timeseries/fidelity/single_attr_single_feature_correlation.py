import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import distribution_similarity


class SingleAttrSingleFeatureCorrelation(TimeSeriesMetric):
    """A metadata attribute a can take different values v. The generated time series for a feature f may depend on the value v of metadata attribute a. This metric checks that correlation qualitatively by fixing the value v, and comparing the distribution of the time series for feature f between real and synthetic data. """

    name = "Correlation between a metadata attribute a and a time series feature f"
    goal = Goal.MINIMIZE

    @classmethod
    def compute(
            cls, real_data, synthetic_data,
            metadata=None, entity_columns=None,
            target=None, configs=None):
        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns)

        if not all(isinstance(s, str) for s in target):
            raise ValueError(
                "target has to be a list of strings where each string specifies an attribute column.")
        assert len(target) == 2, \
            "`target` is expected to be a list including two elements representing one attribute column and one feature column."

        attribute_cols, feature_cols = cls._get_attribute_feature_cols(metadata)
        attr_name, feature_name = target[0], target[1]

        assert attr_name in attribute_cols, \
            f"{attr_name} is not an attribute column."
        assert feature_name in feature_cols, \
            f"{feature_name} is not a feature column."
        assert metadata['fields'][attr_name]['type'] == 'categorical', \
            f"attribute needs to be a categorical variable"

        if metadata['fields'][feature_name]['type'] in ['numerical', 'datetime']:
            cls.min_value = 0.0
            cls.max_value = float("inf")
        elif metadata['fields'][feature_name]['type'] in ['categorical']:
            cls.min_value = 0.0
            cls.max_value = 1.0

        scores = {}
        for v in set(real_data[attr_name]):
            f_real = real_data[real_data[attr_name] == v
                               ][feature_name].to_numpy().reshape(-1, 1)
            f_syn = synthetic_data[synthetic_data[attr_name] == v
                                   ][feature_name].to_numpy().reshape(-1, 1)
            scores[f"{attr_name}: {v}"] = distribution_similarity(
                real_data=f_real,
                synthetic_data=f_syn,
                column_names=[feature_name],
                data_type=[metadata['fields'][feature_name]['type']],
                comparison_type=getattr(configs, 'comparison_type', 'both'),
                categorical_mapping=True
            )

        return scores
