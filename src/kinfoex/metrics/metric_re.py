from typing import Literal

from .re_score.re_score import re_score, ALL
from .metric import ComputeMetrics


class ComputeMetricsForRelationExtraction(ComputeMetrics):

    def __init__(
        self, model_labels: list[str], mode: Literal["strict", "boundaries"] = "strict"
    ) -> None:
        self.mode = mode
        self.model_labels = model_labels

    def compute_metrics(self, p):

        def extract_metrics(metrics):
            re_metrics = {}
            for key in metrics:
                if key == ALL:
                    re_metrics["overall_precision"] = metrics[key]["p"]
                    re_metrics["overall_recall"] = metrics[key]["r"]
                    re_metrics["overall_f1"] = metrics[key]["f1"]
                    re_metrics["macro_precision"] = metrics[key]["macro_p"]
                    re_metrics["macro_recall"] = metrics[key]["macro_r"]
                    re_metrics["macro_f1"] = metrics[key]["macro_f1"]
                    re_metrics["total_n_relations"] = metrics[key]["n_relations"]

                else:
                    re_metrics[f"{key}_precision"] = metrics[key]["p"]
                    re_metrics[f"{key}_recall"] = metrics[key]["r"]
                    re_metrics[f"{key}_f1"] = metrics[key]["f1"]
                    re_metrics[f"{key}_n_relations"] = metrics[key]["n_relations"]
            return re_metrics

        pred_relations, gt_relations = p
        metrics = re_score(
            pred_relations, gt_relations, self.model_labels, mode=self.mode
        )
        metrics = extract_metrics(metrics)

        return metrics
