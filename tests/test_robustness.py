"""
Unit Tests for SafeTV Environmental Robustness Suite.
Validates perturbation transformations and evaluation outputs.
"""

import unittest
import numpy as np

from safetv.robustness import EnvironmentalDegradations, RobustnessEvaluator


class TestRobustnessSuite(unittest.TestCase):
    """Unit tests for surveillance degradation transformations."""

    def test_perturbation_shapes_and_types(self):
        """Validates that perturbations preserve frame dimensions and data type."""
        dummy_frame = np.full((120, 160, 3), fill_value=128, dtype=np.uint8)

        dark = EnvironmentalDegradations.poor_lighting(dummy_frame)
        self.assertEqual(dark.shape, dummy_frame.shape)
        self.assertEqual(dark.dtype, np.uint8)
        self.assertLess(np.mean(dark), np.mean(dummy_frame))

        low_res = EnvironmentalDegradations.low_resolution(dummy_frame, 32, 24)
        self.assertEqual(low_res.shape, dummy_frame.shape)
        self.assertEqual(low_res.dtype, np.uint8)

        blur = EnvironmentalDegradations.motion_blur(dummy_frame)
        self.assertEqual(blur.shape, dummy_frame.shape)

        noisy = EnvironmentalDegradations.sensor_noise(dummy_frame)
        self.assertEqual(noisy.shape, dummy_frame.shape)
        self.assertEqual(noisy.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
