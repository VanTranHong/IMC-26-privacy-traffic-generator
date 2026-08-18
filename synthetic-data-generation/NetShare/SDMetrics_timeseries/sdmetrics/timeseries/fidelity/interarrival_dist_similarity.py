import numpy as np
import pandas as pd

from sdmetrics.goal import Goal
from sdmetrics.timeseries.base import TimeSeriesMetric
from sdmetrics.timeseries.utils import distribution_similarity


class InterarrivalDistSimilarity(TimeSeriesMetric):
    """Checks the inter-arrival time distribution between records (compared to same distribution in real data)"""

    name = "Interarrival distributional similarity"
    goal = Goal.MINIMIZE
    min_value = 0.0
    max_value = float("inf")

    @classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None, configs=None):
        _, entity_columns = cls._validate_inputs(
            real_data, synthetic_data, metadata, entity_columns
        )
        attribute_cols, feature_cols = cls._get_attribute_feature_cols(metadata)

        # By SDV, metadata["sequence_index"] is the column name used to order the rows in the table
        column_sequence_index = metadata["sequence_index"]
        # Convert datetime to unix timestamp (unit: second) in-place
        if metadata["fields"][column_sequence_index]["type"] == "datetime":
            real_data[column_sequence_index] = (pd.to_datetime(
                real_data[column_sequence_index]).astype(int) / 10**9)
            synthetic_data[column_sequence_index] = (pd.to_datetime(
                synthetic_data[column_sequence_index]).astype(int) / 10**9)

        real_gk = real_data.groupby(attribute_cols)
        real_interarrival_within_flow_list = []
        synthetic_gk = synthetic_data.groupby(attribute_cols)
        synthetic_interarrival_within_flow_list = []

        for group_name, df_group in real_gk:
            real_interarrival_within_flow_list += list(
                np.diff(df_group[column_sequence_index])
            )
        real_interarrival_within_flow_list = np.asarray(
            real_interarrival_within_flow_list
        ).reshape(-1, 1)
        for group_name, df_group in synthetic_gk:
            synthetic_interarrival_within_flow_list += list(
                np.diff(df_group[column_sequence_index])
            )
        synthetic_interarrival_within_flow_list = np.asarray(
            synthetic_interarrival_within_flow_list
        ).reshape(-1, 1)

        return distribution_similarity(
            real_data=real_interarrival_within_flow_list,
            synthetic_data=synthetic_interarrival_within_flow_list,
            column_names=["interarrival"],
            data_type=["numerical"],
            comparison_type=getattr(configs, "comparison_type", "both"),
            categorical_mapping=True
        )
