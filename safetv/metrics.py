"""
Evaluation metrics module for SafeTV video violence detection engine.
Implements window-level and event-level evaluation including Precision, Recall, F1,
Specificity, Balanced Accuracy, False Alarm Rate (FAR), and temporal IoU.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class EvaluationReport:
    """Summary dataclass of performance evaluation metrics."""
    total_samples: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    balanced_accuracy: float
    false_alarm_rate: float
    mean_detection_latency_sec: Optional[float] = None
    mean_temporal_iou: Optional[float] = None

    def to_dict(self) -> Dict[str, Union[int, float, None]]:
        return {
            "total_samples": self.total_samples,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "f1_score": round(self.f1_score, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "false_alarm_rate": round(self.false_alarm_rate, 4),
            "mean_detection_latency_sec": round(self.mean_detection_latency_sec, 3) if self.mean_detection_latency_sec is not None else None,
            "mean_temporal_iou": round(self.mean_temporal_iou, 3) if self.mean_temporal_iou is not None else None,
        }


class ViolenceDetectionEvaluator:
    """
    Computes rigorous classification and temporal metrics on video sequences.
    """

    @staticmethod
    def compute_window_metrics(
        y_true: List[int],
        y_pred: List[int],
        y_prob: Optional[List[float]] = None
    ) -> EvaluationReport:
        """
        Evaluate binary window-level predictions.
        0: Normal activity, 1: Violence.
        """
        y_true_arr = np.array(y_true, dtype=int)
        y_pred_arr = np.array(y_pred, dtype=int)

        tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
        fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
        tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
        fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
        total = len(y_true_arr)

        accuracy = (tp + tn) / max(1, total)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        f1 = 2 * (precision * recall) / max(1e-7, precision + recall)
        balanced_acc = (recall + specificity) / 2.0
        false_alarm_rate = fp / max(1, tn + fp)

        return EvaluationReport(
            total_samples=total,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            specificity=specificity,
            f1_score=f1,
            balanced_accuracy=balanced_acc,
            false_alarm_rate=false_alarm_rate,
        )

    @staticmethod
    def compute_temporal_iou(
        pred_interval: Tuple[float, float],
        gt_interval: Tuple[float, float]
    ) -> float:
        """
        Compute temporal Intersection over Union (tIoU) between predicted interval
        (p_start, p_end) and ground truth interval (g_start, g_end).
        """
        p_start, p_end = pred_interval
        g_start, g_end = gt_interval

        intersection = max(0.0, min(p_end, g_end) - max(p_start, g_start))
        union = max(0.0, max(p_end, g_end) - min(p_start, g_start))

        return float(intersection / union) if union > 0 else 0.0
