"""Base Time Series metric class."""

from operator import attrgetter

from sdmetrics.base import BaseMetric
from sdmetrics.utils import get_columns_from_metadata
from sdmetrics.goal import Goal

import numpy as np


class TimeSeriesMetric(BaseMetric):
    """Base class for metrics that apply to time series.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = None
    goal = None
    min_value = None
    max_value = None

    _DTYPES_TO_TYPES = {
        'i': {
            'type': 'numerical',
            'subtype': 'integer',
        },
        'f': {
            'type': 'numerical',
            'subtype': 'float',
        },
        'O': {
            'type': 'categorical',
        },
        'b': {
            'type': 'boolean',
        },
        'M': {
            'type': 'datetime',
        }
    }

    @classmethod
    def _insert_best_worst_score_metrics_output(cls, metrics_output):
        if cls.goal == Goal.MINIMIZE:
            cls.best_score = cls.min_value
            cls.worst_score = cls.max_value
        elif cls.goal == Goal.MAXIMIZE:
            cls.best_score = cls.max_value
            cls.worst_score = cls.min_value
        else:
            raise ValueError("Non-compatible goal.")

        if isinstance(metrics_output, list):
            return [(metrics_output[0], cls.best_score, cls.worst_score), metrics_output[1]] if len(metrics_output) > 1 else [(metrics_output[0], cls.best_score, cls.worst_score)]
        elif isinstance(metrics_output, dict):
            for k, v in metrics_output.items():
                metrics_output[k] = cls._insert_best_worst_score_metrics_output(
                    v)

            return metrics_output

    @classmethod
    def _get_attribute_feature_cols(cls, metadata):
        attribute_cols = metadata['entity_columns'] + \
            metadata['context_columns']
        feature_cols = list(
            set(list(metadata['fields'].keys())) - set(attribute_cols))

        return attribute_cols, feature_cols

    @classmethod
    def _load_attribute_feature(cls, data, metadata=None, entity_columns=None):
        '''Construct `data_attribute` and `data_feature`'''
        attribute_cols = metadata['entity_columns'] + metadata['context_columns']
        feature_cols = list(set(data.columns) - set(attribute_cols))

        gk = data.groupby(attribute_cols)
        # data_attribute
        data_attribute = np.array(list(gk.groups.keys()))

        # data_feature
        max_sample_len = max(gk.size().values)
        data_feature = []
        for group_name, df_group in gk:
            df_numpy = np.pad(
                np.expand_dims(
                    df_group[feature_cols].to_numpy(), axis=0),
                pad_width=(
                    (0, 0), (0, max_sample_len-len(df_group)), (0, 0)),
                mode="constant",
                constant_values=0
            )
            data_feature.append(df_numpy)
        data_feature = np.concatenate(data_feature, axis=0)

        # data_gen_flag: indicating timeseries with unequal length
        data_gen_flag = (np.count_nonzero(
            data_feature, axis=2) > 0).astype(int)

        return data_attribute, data_feature, data_gen_flag

    @ classmethod
    def _validate_inputs(
            cls, real_data, synthetic_data, metadata=None, entity_columns=None):
        if set(real_data.columns) != set(synthetic_data.columns):
            raise ValueError(
                '`real_data` and `synthetic_data` must have the same columns')

        if metadata is not None:
            if not isinstance(metadata, dict):
                metadata = metadata.to_dict()

            fields = get_columns_from_metadata(metadata)
            for column in real_data.columns:
                if column not in fields:
                    raise ValueError(f'Column {column} not found in metadata')

            for field in fields.keys():
                if field not in real_data.columns:
                    raise ValueError(f'Field {field} not found in data')

        else:
            dtype_kinds = real_data.dtypes.apply(attrgetter('kind'))
            metadata = {'fields': dtype_kinds.apply(
                cls._DTYPES_TO_TYPES.get).to_dict()}

        entity_columns = metadata.get('entity_columns', entity_columns or [])

        return metadata, entity_columns

    @ classmethod
    def compute(cls, real_data, synthetic_data, metadata=None,
                entity_columns=None):
        """Compute this metric.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as a pandas.DataFrame.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a pandas.DataFrame.
            metadata (dict):
                TimeSeries metadata dict. If not passed, it is build based on the
                real_data fields and dtypes.
            entity_columns (list[str]):
                Names of the columns which identify different time series
                sequences.

        Returns:
            Union[float, tuple[float]]:
                Metric output: [(cur_score, best_score, worst_score), fig]
        """
        raise NotImplementedError()
