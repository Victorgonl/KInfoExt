from dataclasses import dataclass
from typing import Union

import numpy

from .seqeval.seqeval import Seqeval
from .metric import ComputeMetrics


class ComputeMetricsForTokenClassification(ComputeMetrics):

    def __init__(self, model_labels, tag_format) -> None:
        self.model_labels = model_labels
        self.tag_format = tag_format
        self.seqeval: Seqeval = Seqeval()

    def compute_metrics(self, p):

        def extract_metrics(m):
            metrics = {}
            for k in m.keys():
                if type(m[k]) is dict:
                    for j in m[k].keys():
                        metrics[k + "_" + j] = m[k][j]
                else:
                    metrics[k] = m[k]
            return metrics

        predictions, labels = p
        predictions = numpy.argmax(predictions, axis=2)
        all_predictions = []
        all_labels = []
        for prediction, label in zip(predictions, labels):
            for predicted_idx, label_idx in zip(prediction, label):
                if label_idx == -100:
                    continue
                all_predictions.append(self.model_labels[predicted_idx])
                all_labels.append(self.model_labels[label_idx])
        metrics = self.seqeval.compute(
            predictions=[all_predictions],
            references=[all_labels],
            scheme=self.tag_format,
        )
        if metrics is not None:
            return extract_metrics(metrics)
