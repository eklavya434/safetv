"""
SafeTV Probability Calibration & Evaluation Module.
Computes Expected Calibration Error (ECE), Brier Score, and Reliability Diagrams.
Implements Temperature Scaling for neural network confidence calibration.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalibrationMetrics:
    """Quantitative confidence calibration metrics."""
    brier_score: float
    expected_calibration_error: float
    max_calibration_error: float
    num_samples: int
    num_bins: int
    bin_confidences: List[float]
    bin_accuracies: List[float]
    bin_counts: List[int]

    def to_dict(self) -> dict:
        return asdict(self)


class ProbabilityCalibrator:
    """
    Evaluates probability calibration and applies temperature scaling to uncalibrated logits.
    """

    def __init__(self, temperature: float = 1.0, calibration_file: Optional[str] = None):
        self.temperature: float = max(0.01, float(temperature))
        self.calibration_file = calibration_file or "models/violence_detector/calibration.json"
        self._load_calibration()

    def _load_calibration(self) -> None:
        """Load temperature parameter if available."""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.temperature = max(0.01, float(data.get("temperature", 1.0)))
            except Exception:
                pass

    def save_calibration(self, output_path: Optional[str] = None) -> None:
        """Save temperature scaling parameters."""
        path = output_path or self.calibration_file
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "temperature": round(self.temperature, 4),
                "method": "temperature_scaling",
                "calibration_date": str(np.datetime64("now")),
            }, f, indent=2)

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """
        Scales logits by temperature and applies softmax:
        p_i = exp(z_i / T) / sum(exp(z_j / T))
        """
        scaled = logits / self.temperature
        # Numerically stable softmax
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)

    @staticmethod
    def compute_brier_score(y_true: List[int], y_prob: List[float]) -> float:
        """
        Calculates Brier Score: Mean Squared Error between predicted probabilities and binary outcomes.
        Lower is better (0.0 = perfect calibration and discrimination).
        """
        y_t = np.array(y_true, dtype=np.float32)
        y_p = np.array(y_prob, dtype=np.float32)
        return float(np.mean(np.square(y_p - y_t)))

    @staticmethod
    def compute_ece(
        y_true: List[int],
        y_prob: List[float],
        num_bins: int = 10
    ) -> CalibrationMetrics:
        """
        Calculates Expected Calibration Error (ECE) and builds reliability diagram bins.
        """
        y_t = np.array(y_true, dtype=np.int32)
        y_p = np.array(y_prob, dtype=np.float32)
        n = len(y_t)

        if n == 0:
            return CalibrationMetrics(0.0, 0.0, 0.0, 0, num_bins, [], [], [])

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        ece = 0.0
        mce = 0.0

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # In bin condition
            if i == num_bins - 1:
                in_bin = (y_p >= bin_lower) & (y_p <= bin_upper)
            else:
                in_bin = (y_p >= bin_lower) & (y_p < bin_upper)

            count = int(np.sum(in_bin))
            bin_counts.append(count)

            if count > 0:
                bin_acc = float(np.mean(y_t[in_bin]))
                bin_conf = float(np.mean(y_p[in_bin]))
                diff = abs(bin_acc - bin_conf)
                ece += (count / n) * diff
                mce = max(mce, diff)
                bin_accuracies.append(round(bin_acc, 4))
                bin_confidences.append(round(bin_conf, 4))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(round((bin_lower + bin_upper) / 2.0, 4))

        brier = ProbabilityCalibrator.compute_brier_score(y_true, y_prob)

        return CalibrationMetrics(
            brier_score=round(brier, 4),
            expected_calibration_error=round(float(ece), 4),
            max_calibration_error=round(float(mce), 4),
            num_samples=n,
            num_bins=num_bins,
            bin_confidences=bin_confidences,
            bin_accuracies=bin_accuracies,
            bin_counts=bin_counts,
        )
