"""
SafeTV Environmental Robustness & Stress-Testing Suite.
Evaluates model stability under degraded surveillance conditions:
1. Low illumination / Night vision (gamma degradation, exposure drop)
2. Low resolution / Heavy compression (spatial downscaling)
3. Motion blur (fast camera panning / jitter)
4. Sensor noise (Gaussian & Salt-and-Pepper noise)
"""

import json
import os
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

from safetv.metrics import ViolenceDetectionEvaluator
from safetv.pipeline import SafeTVPipeline


class EnvironmentalDegradations:
    """Applies realistic surveillance perturbations to video frames."""

    @staticmethod
    def poor_lighting(frame: np.ndarray, factor: float = 0.35) -> np.ndarray:
        """Simulates poorly lit / night-time surveillance environments."""
        # Non-linear gamma drop + brightness scaling
        inv_gamma = 1.8
        table = np.array([((i / 255.0) ** inv_gamma) * 255 * factor for i in range(256)]).astype(np.uint8)
        return cv2.LUT(frame, table)

    @staticmethod
    def low_resolution(frame: np.ndarray, target_w: int = 48, target_h: int = 36) -> np.ndarray:
        """Simulates low-bandwidth / heavy downscaling compression."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def motion_blur(frame: np.ndarray, kernel_size: int = 9) -> np.ndarray:
        """Simulates camera shake or rapid panning."""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(frame, -1, kernel)

    @staticmethod
    def sensor_noise(frame: np.ndarray, sigma: float = 25.0) -> np.ndarray:
        """Simulates thermal/CMOS sensor noise under low signal conditions."""
        noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy


class RobustnessEvaluator:
    """
    Evaluates SafeTV pipeline performance across pristine and degraded surveillance conditions.
    """

    def __init__(self, pipeline: SafeTVPipeline = None):
        self.pipeline = pipeline or SafeTVPipeline()

    def evaluate_scenario(
        self,
        frames: List[np.ndarray],
        fps: float,
        perturbation_fn: Callable[[np.ndarray], np.ndarray],
        scenario_name: str
    ) -> Dict:
        """Run inference on perturbed video frames and collect detection metrics."""
        perturbed = [perturbation_fn(f.copy()) for f in frames]
        report = self.pipeline.process_frames(perturbed, fps=fps, source_name=scenario_name)
        return {
            "scenario": scenario_name,
            "peak_probability": round(report.peak_violence_prob, 4),
            "mean_probability": round(report.mean_violence_prob, 4),
            "incidents_detected": report.incidents_detected,
            "windows_processed": report.total_windows_processed,
            "execution_time_sec": round(report.execution_time_sec, 3),
        }

    def run_full_stress_test(
        self,
        violence_frames: List[np.ndarray],
        normal_frames: List[np.ndarray],
        fps: float = 30.0
    ) -> Dict:
        """
        Runs comprehensive stress test comparing baseline performance against
        poor lighting, low resolution, motion blur, and sensor noise.
        """
        stress_tests = {
            "baseline": lambda f: f,
            "poor_lighting_35pct": EnvironmentalDegradations.poor_lighting,
            "low_resolution_48x36": EnvironmentalDegradations.low_resolution,
            "motion_blur_k9": EnvironmentalDegradations.motion_blur,
            "sensor_noise_sigma25": EnvironmentalDegradations.sensor_noise,
        }

        results = {
            "violence_detection_under_stress": {},
            "normal_rejection_under_stress": {},
        }

        # 1. Evaluate violence detection persistence under stress
        for name, fn in stress_tests.items():
            res = self.evaluate_scenario(violence_frames, fps, fn, f"violence_{name}")
            results["violence_detection_under_stress"][name] = res

        # 2. Evaluate false alarm rate resilience under stress
        for name, fn in stress_tests.items():
            res = self.evaluate_scenario(normal_frames, fps, fn, f"normal_{name}")
            results["normal_rejection_under_stress"][name] = res

        return results
