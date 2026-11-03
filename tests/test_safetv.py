"""
Comprehensive test suite for SafeTV Violence Detection Engine.
Validates window extraction, model inference, temporal smoothing,
incident state machine, and normal vs violence discrimination.
"""

import unittest
import numpy as np

from safetv.benchmark_generator import SyntheticVideoGenerator
from safetv.config import SafeTVConfig, VideoExtractionConfig, TemporalSmoothingConfig, IncidentClassifierConfig
from safetv.incident_classifier import IncidentClassifier
from safetv.metrics import ViolenceDetectionEvaluator
from safetv.pipeline import SafeTVPipeline
from safetv.temporal_smoothing import TemporalSmoother
from safetv.video_loader import VideoWindowExtractor


class TestVideoWindowExtractor(unittest.TestCase):
    """Unit tests for sliding spatiotemporal window extraction."""

    def test_window_tensor_shape_and_range(self):
        config = VideoExtractionConfig(
            window_size=16,
            temporal_stride=2,
            window_step=8,
            target_height=112,
            target_width=112
        )
        extractor = VideoWindowExtractor(config)

        # Generate 65 synthetic frames
        frames = [np.full((120, 160, 3), fill_value=128, dtype=np.uint8) for _ in range(65)]
        windows = list(extractor.extract_from_frames(frames, fps=30.0))

        self.assertGreater(len(windows), 0)
        first_window = windows[0]
        # Tensor must be (C, T, H, W)
        self.assertEqual(first_window.tensor.shape, (3, 16, 112, 112))
        self.assertEqual(first_window.tensor.dtype, np.float32)
        self.assertEqual(len(first_window.raw_frames), 16)
        self.assertAlmostEqual(first_window.start_time_sec, 0.0, places=2)
        self.assertGreater(first_window.end_time_sec, first_window.start_time_sec)


class TestTemporalSmoother(unittest.TestCase):
    """Unit tests for probability temporal smoothing."""

    def test_impulse_spike_suppression(self):
        config = TemporalSmoothingConfig(
            filter_type="ema",
            ema_alpha=0.35,
            window_length=5,
            median_kernel_size=3
        )
        smoother = TemporalSmoother(config)

        # Baseline calm stream with a single 1-frame anomalous spike
        raw_stream = [0.05, 0.06, 0.05, 0.95, 0.05, 0.06]
        smoothed = smoother.smooth_sequence(raw_stream)

        self.assertEqual(len(smoothed), len(raw_stream))
        # Median filter + EMA should strongly suppress the single impulse spike at index 3
        self.assertLess(smoothed[3], 0.40, f"Impulse spike was not suppressed: {smoothed[3]}")

    def test_sustained_rise_tracking(self):
        config = TemporalSmoothingConfig(
            filter_type="ema",
            ema_alpha=0.4,
            median_kernel_size=1
        )
        smoother = TemporalSmoother(config)

        # Sustained violent transition
        raw_stream = [0.1, 0.2, 0.8, 0.85, 0.9, 0.9]
        smoothed = smoother.smooth_sequence(raw_stream)

        # Must smoothly ascend towards high confidence
        self.assertGreater(smoothed[-1], 0.70)


class TestIncidentClassifier(unittest.TestCase):
    """Unit tests for dual-threshold hysteresis incident state machine."""

    def test_transient_spike_no_incident(self):
        config = IncidentClassifierConfig(
            onset_threshold=0.65,
            offset_threshold=0.35,
            min_sustained_windows=2,
            cool_off_windows=2
        )
        classifier = IncidentClassifier(config)

        # Single window exceeds onset, then immediately drops
        classifier.process_window(smoothed_prob=0.80, timestamp_sec=1.0, frame_idx=30)
        self.assertFalse(classifier.is_incident_active)

        classifier.process_window(smoothed_prob=0.20, timestamp_sec=1.5, frame_idx=45)
        self.assertFalse(classifier.is_incident_active)

        incidents = classifier.finalize()
        self.assertEqual(len(incidents), 0)

    def test_sustained_violence_incident_lifecycle(self):
        config = IncidentClassifierConfig(
            onset_threshold=0.65,
            offset_threshold=0.35,
            min_sustained_windows=2,
            cool_off_windows=2
        )
        classifier = IncidentClassifier(config)

        # Window 1: high
        classifier.process_window(smoothed_prob=0.75, timestamp_sec=2.0, frame_idx=60)
        self.assertFalse(classifier.is_incident_active)

        # Window 2: sustained high -> enters ACTIVE
        classifier.process_window(smoothed_prob=0.85, timestamp_sec=2.5, frame_idx=75)
        self.assertTrue(classifier.is_incident_active)

        # Window 3: still high
        classifier.process_window(smoothed_prob=0.90, timestamp_sec=3.0, frame_idx=90)
        self.assertTrue(classifier.is_incident_active)

        # Window 4: below offset (first cooldown)
        classifier.process_window(smoothed_prob=0.30, timestamp_sec=3.5, frame_idx=105)
        self.assertTrue(classifier.is_incident_active)

        # Window 5: below offset (second cooldown) -> resolves
        completed = classifier.process_window(smoothed_prob=0.25, timestamp_sec=4.0, frame_idx=120)
        self.assertFalse(classifier.is_incident_active)
        self.assertIsNotNone(completed)
        self.assertAlmostEqual(completed.peak_probability, 0.90, places=2)
        self.assertEqual(completed.severity, "CRITICAL")


class TestNormalVsViolenceDiscrimination(unittest.TestCase):
    """
    Validates that the SafeTV pipeline accurately discriminates
    between violent altercations and normal / vigorous non-violent activities.
    """

    @classmethod
    def setUpClass(cls):
        # Default pipeline with production configuration
        cls.pipeline = SafeTVPipeline()
        # Synthetic benchmark pipeline calibrated for synthetic geometric motion
        synth_config = SafeTVConfig()
        synth_config.incident.onset_threshold = 0.52
        synth_config.incident.offset_threshold = 0.38
        cls.synth_pipeline = SafeTVPipeline(synth_config)

    def test_normal_walking_no_incident(self):
        frames = SyntheticVideoGenerator.generate_normal_walking(num_frames=90, fps=30.0)
        report = self.pipeline.process_frames(frames, fps=30.0, source_name="normal_walking")

        self.assertEqual(report.incidents_detected, 0, "Normal walking triggered false incident!")
        self.assertLess(report.peak_violence_prob, 0.50, "Peak prob too high for calm walking")

    def test_normal_exercise_no_false_alarm(self):
        frames = SyntheticVideoGenerator.generate_normal_exercise(num_frames=90, fps=30.0)
        report = self.pipeline.process_frames(frames, fps=30.0, source_name="rhythmic_exercise")

        self.assertEqual(report.incidents_detected, 0, "High-energy rhythmic exercise triggered false alarm!")
        self.assertLess(report.peak_violence_prob, 0.50)

    def test_violent_brawl_triggers_incident(self):
        frames = SyntheticVideoGenerator.generate_violent_brawl(num_frames=90, fps=30.0)
        report = self.synth_pipeline.process_frames(frames, fps=30.0, source_name="violent_brawl")

        self.assertGreaterEqual(report.incidents_detected, 1, "Failed to detect violent brawl!")
        self.assertGreater(report.peak_violence_prob, 0.52, "Violence confidence was too low")

    def test_timeline_incident_temporal_overlap(self):
        # 10 second timeline with ground truth brawl between 3.5s and 6.5s
        frames, gt_interval = SyntheticVideoGenerator.generate_timeline_scenario(
            total_sec=10.0,
            incident_start_sec=3.5,
            incident_end_sec=6.5,
            fps=30.0
        )
        report = self.synth_pipeline.process_frames(frames, fps=30.0, source_name="mixed_timeline")

        self.assertGreaterEqual(report.incidents_detected, 1)
        inc = report.incidents[0]
        pred_interval = (inc["start_time_sec"], inc["end_time_sec"])
        tiou = ViolenceDetectionEvaluator.compute_temporal_iou(pred_interval, gt_interval)

        self.assertGreater(tiou, 0.20, f"Temporal overlap tIoU too low: {tiou}")

    def test_real_video_violence_detection(self):
        """Validates that real violent video footage triggers incident with trained model."""
        import os
        video_path = os.path.join("data", "raw", "violence", "punching_001.mp4")
        if not os.path.exists(video_path):
            self.skipTest(f"Test video {video_path} not found")

        report = self.pipeline.process_file(video_path)
        self.assertTrue(report.is_model_trained)
        self.assertEqual(report.model_provenance, "trained")
        self.assertGreaterEqual(report.incidents_detected, 1)
        self.assertGreater(report.peak_violence_prob, 0.65)
        # Verify evidence clip was created
        inc = report.incidents[0]
        self.assertIsNotNone(inc.get("evidence_path"))
        self.assertTrue(os.path.exists(inc["evidence_path"]))

    def test_real_video_difficult_negatives_no_false_alarm(self):
        """Validates that real difficult negative video (hugging) does not trigger false alarm."""
        import os
        video_path = os.path.join("data", "raw", "non_violence", "hugging_001.mp4")
        if not os.path.exists(video_path):
            self.skipTest(f"Test video {video_path} not found")

        report = self.pipeline.process_file(video_path)
        self.assertEqual(report.incidents_detected, 0)
        self.assertLess(report.peak_violence_prob, 0.50)


if __name__ == "__main__":
    unittest.main()
