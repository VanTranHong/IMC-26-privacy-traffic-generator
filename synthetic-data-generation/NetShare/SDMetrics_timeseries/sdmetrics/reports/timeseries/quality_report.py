"""Timeseries quality report"""
import os
import sys
import pickle
import random
import warnings
import pkg_resources
import dash
import inspect
import importlib
import pprint


from dash import dcc, html
from dash.dependencies import Input, Output
from tqdm import tqdm
from config_io import Config
from collections import OrderedDict


class QualityReport:
    def __init__(self, config_file=None, config_dict=None):
        if config_dict is not None:
            self._config = Config(config_dict)
        else:
            self._config = Config.load_from_file(
                config_file, default="config_quality_report.json",
                default_search_paths=[os.path.dirname(
                    inspect.getfile(self.__class__))],)
        self.graph_idx = 0

    # Different metrics have different depths
    # E.g., `single_attr_dist` has depth=2, `interarrival` has depth=1
    # TODO: prettier layout
    def _traverse_metrics_dict(self, metrics_dict, html_children):
        for main_metric, scores in metrics_dict.items():
            html_children.append(html.Div(html.H2(main_metric)))
            if isinstance(scores, list):
                score = scores[0]
                if len(scores) > 1 and scores[1] is not None:  # valid plot
                    fig = scores[1]
                    html_children.append(
                        html.Div(
                            [
                                html.Div(
                                    "score: {:.3f} (best: {:.3f}, worst: {:.3f})".format(
                                        score[0], score[1], score[2]
                                    )
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            # TODO: better graph index
                                            id=f"graph-{self.graph_idx}",
                                            figure=fig,
                                            style={"width": "100vh"},
                                        )
                                    ]
                                ),
                            ]
                        )
                    )
                    self.graph_idx += 1
                else:
                    html_children.append(
                        html.Div(
                            "score: {:.3f} (best: {:.3f}, worst: {:.3f})".format(
                                score[0], score[1], score[2]
                            )
                        )
                    )
            else:
                self._traverse_metrics_dict(scores, html_children)

    def visualize(self):
        app = dash.Dash(__name__)
        html_children = []
        for metric_type, metrics_dict in self.dict_metric_scores.items():
            html_children.append(
                html.Div(
                    html.H1(children=metric_type),
                )
            )
            self._traverse_metrics_dict(metrics_dict, html_children)

        app.layout = html.Div(children=html_children)
        app.run_server(debug=False)

    def generate(self, real_data, synthetic_data, metadata, out=sys.stdout):
        self.dict_metric_scores = OrderedDict()

        for metric_type, metrics in self._config["metrics"].items():
            # fidelity/privacy
            metric_module = importlib.import_module(
                f"sdmetrics.timeseries.{metric_type}"
            )
            self.dict_metric_scores[metric_type] = OrderedDict()
            for metric_dict in metrics:
                metric_name = list(metric_dict.keys())[0]
                metric_config = list(metric_dict.values())[0]
                metric_class = getattr(metric_module, metric_config["class"])
                metric_class_instance = metric_class()

                # Metrics that do not have `target` (e.g., session length)
                if "target_list" not in metric_config:
                    _real_data = real_data.copy(deep=True)
                    _synthetic_data = synthetic_data.copy(deep=True)
                    self.dict_metric_scores[metric_type][
                        metric_name
                    ] = metric_class_instance._insert_best_worst_score_metrics_output(
                        metric_class_instance.compute(
                            _real_data, _synthetic_data, metadata,
                            configs=getattr(metric_config, "configs", None)
                        )
                    )

                # Metrics that have `target` (e.g., single attribute distributional similarity)
                else:
                    if metric_name not in self.dict_metric_scores[metric_type]:
                        self.dict_metric_scores[metric_type][metric_name] = OrderedDict(
                        )
                    for target in metric_config["target_list"]:
                        _real_data = real_data.copy(deep=True)
                        _synthetic_data = synthetic_data.copy(deep=True)
                        self.dict_metric_scores[metric_type][metric_name][
                            str(target)
                        ] = metric_class_instance._insert_best_worst_score_metrics_output(
                            metric_class_instance.compute(
                                _real_data, _synthetic_data, metadata, target=target,
                                configs=getattr(metric_config, "configs", None)
                            )
                        )

    def save(self, filepath):
        """Save this report instance to the given path using pickle.

        Args:
            filepath (str):
                The path to the file where the report instance will be serialized.
        """
        self._package_version = pkg_resources.get_distribution(
            "sdmetrics").version

        with open(filepath, "wb") as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, filepath):
        """Load a ``QualityReport`` instance from a given path.

        Args:
            filepath (str):
                The path to the file where the report is stored.

        Returns:
            QualityReort:
                The loaded quality report instance.
        """
        current_version = pkg_resources.get_distribution("sdmetrics").version

        with open(filepath, "rb") as f:
            report = pickle.load(f)
            if current_version != report._package_version:
                warnings.warn(
                    f"The report was created using SDMetrics version `{report._package_version}` "
                    f"but you are currently using version `{current_version}`. "
                    "Some features may not work as intended.")

            return report
