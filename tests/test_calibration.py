"""
Unit tests for SafeTV Probability Calibration & Evaluation.
Validates Brier score calculation, Expected Calibration Error (ECE),
and Temperature Scaling logit transformation.
"""

import unittest
import numpy as np

from safetv.calibration import ProbabilityCalibrator


class TestProbabilityCalibration(unittest.TestCase):
    """Unit tests for ProbabilityCalibrator."""

    def test_brier_score_perfect_and_worst(self):
        # Perfect predictions: Brier score must be 0.0
        y_true = [1, 0, 1, 0]
        y_prob_perfect = [1.0, 0.0, 1.0, 0.0]
        brier_perfect = ProbabilityCalibrator.compute_brier_score(y_true, y_prob_perfect)
        self.assertAlmostEqual(brier_perfect, 0.0, places=4)

        # Completely wrong predictions: Brier score must be 1.0
        y_prob_worst = [0.0, 1.0, 0.0, 1.0]
        brier_worst = ProbabilityCalibrator.compute_brier_score(y_true, y_prob_worst)
        self.assertAlmostEqual(brier_worst, 1.0, places=4)

        # Uncertain 0.5 predictions: Brier score must be 0.25
        y_prob_half = [0.5, 0.5, 0.5, 0.5]
        brier_half = ProbabilityCalibrator.compute_brier_score(y_true, y_prob_half)
        self.assertAlmostEqual(brier_half, 0.25, places=4)

    def test_ece_metric_computation(self):
        # 10 samples
        y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        y_prob = [0.1, 0.15, 0.2, 0.05, 0.1, 0.85, 0.9, 0.95, 0.8, 0.75]

        metrics = ProbabilityCalibrator.compute_ece(y_true, y_prob, num_bins=5)
        self.assertEqual(metrics.num_samples, 10)
        self.assertEqual(metrics.num_bins, 5)
        self.assertGreaterEqual(metrics.expected_calibration_error, 0.0)
        self.assertLessEqual(metrics.expected_calibration_error, 1.0)
        self.assertGreaterEqual(metrics.brier_score, 0.0)
        self.assertLess(metrics.brier_score, 0.10) # well-separated predictions

    def test_temperature_scaling_logits(self):
        calibrator = ProbabilityCalibrator(temperature=2.0)
        logits = np.array([[2.0, -2.0]]) # binary logit vector

        calibrated_probs = calibrator.calibrate_logits(logits)
        self.assertEqual(calibrated_probs.shape, (1, 2))
        self.assertAlmostEqual(float(np.sum(calibrated_probs)), 1.0, places=5)

        # With temperature = 2.0 (softer):
        # original softmax p[0] is ~ 0.982
        # scaled logits [1.0, -1.0] -> softmax p[0] is ~ 0.880
        self.assertLess(calibrated_probs[0, 0], 0.98)
        self.assertGreater(calibrated_probs[0, 0], 0.80)


if __name__ == "__main__":
    unittest.main()
