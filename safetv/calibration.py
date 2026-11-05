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

    def __init__(self, temperature: Optional[float] = None, calibration_file: Optional[str] = None):
        self.calibration_file = calibration_file or "models/violence_detector/calibration.json"
        if temperature is not None:
            self.temperature: float = max(0.01, float(temperature))
        else:
            self.temperature: float = 1.0
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

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Fits optimal temperature scaling parameter T via Negative Log Likelihood (NLL) minimization
        on validation set logits and ground truth labels (Guo et al., 2017).
        """
        logits_arr = np.array(logits, dtype=np.float64)
        labels_arr = np.array(labels, dtype=np.int64)

        if len(logits_arr) == 0 or len(labels_arr) == 0:
            return self.temperature

        def nll_objective(t_val: float) -> float:
            t = max(0.01, float(t_val))
            scaled = logits_arr / t
            # Stable log-softmax: z - max(z) - log(sum(exp(z - max(z))))
            max_z = np.max(scaled, axis=-1, keepdims=True)
            shifted = scaled - max_z
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + 1e-12)
            log_probs = shifted - log_sum_exp

            # Extract log prob for true class
            nll = 0.0
            for idx, target in enumerate(labels_arr):
                nll -= log_probs[idx, target]
            return float(nll / len(labels_arr))

        # Optimize using scipy if installed, otherwise fine golden-section search
        best_t = 1.0
        try:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(nll_objective, bounds=(0.05, 10.0), method="bounded")
            if res.success:
                best_t = float(res.x)
        except ImportError:
            # Golden section search over [0.05, 10.0]
            a, b = 0.05, 10.0
            invphi = (np.sqrt(5) - 1) / 2
            invphi2 = (3 - np.sqrt(5)) / 2
            h = b - a
            c = a + invphi2 * h
            d = a + invphi * h
            yc = nll_objective(c)
            yd = nll_objective(d)

            for _ in range(30):
                if yc < yd:
                    b, d, yd = d, c, yc
                    h = invphi * h
                    c = a + invphi2 * h
                    yc = nll_objective(c)
                else:
                    a, c, yc = c, d, yd
                    h = invphi * h
                    d = a + invphi * h
                    yd = nll_objective(d)

            best_t = float((a + b) / 2.0)

        self.temperature = max(0.01, round(best_t, 4))
        self.save_calibration()
        return self.temperature

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
