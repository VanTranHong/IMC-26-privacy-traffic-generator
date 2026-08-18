import warnings

from sdmetrics.reports.utils import make_discrete_column_plot_single, make_continuous_column_plot_single


class SingleDatasetVisualize():
    def visualize(self, real_data, metadata):
        for field_name, field in metadata['fields'].items():
            if field['type'] == 'categorical':
                make_discrete_column_plot_single(
                    real_column=real_data[field_name],
                    sdtype='categorical').show()
            elif field['type'] == 'numerical':
                make_continuous_column_plot_single(
                    real_column=real_data[field_name],
                    sdtype='numerical'
                ).show()
            else:
                warnings.warn(f"Unsupported field type `{field['type']}`.")
