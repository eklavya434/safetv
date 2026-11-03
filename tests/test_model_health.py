"""
Tests for SafeTV Model Health Check and Provenance Reporting (Phase 7).
"""

import unittest
from fastapi.testclient import TestClient

from safetv.config import ModelConfig
from safetv.model import VideoActionClassifier
from safetv.server import app


class TestModelHealthCheck(unittest.TestCase):
    """Validates health check data structures, integrity checks, and API endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_report_structure(self):
        """Validates that get_health_report returns all Phase 7 required fields."""
        classifier = VideoActionClassifier()
        report = classifier.get_health_report()

        self.assertIn("architecture", report)
        self.assertIn("checkpoint_loaded", report)
        self.assertIn("checkpoint_hash", report)
        self.assertIn("training_provenance", report)
        self.assertIn("class_mapping", report)
        self.assertIn("calibration_status", report)
        self.assertIn("is_suitable_for_evaluation", report)
        self.assertIn("is_production_ready", report)
        self.assertIn("preprocessing_config", report)
        self.assertIn("status_message", report)

        # Check preprocessing details
        prep = report["preprocessing_config"]
        self.assertEqual(prep["temporal_window_frames"], 16)
        self.assertEqual(prep["spatial_resolution"], [112, 112])
        self.assertEqual(prep["channel_order"], "RGB")

        # Check class mapping
        self.assertIn(report["class_mapping"][0], ["non_violence", "normal"])
        self.assertEqual(report["class_mapping"][1], "violence")

    def test_api_model_health_endpoint(self):
        """Validates GET /api/model/health returns 200 with structured health report."""
        response = self.client.get("/api/model/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("architecture", data)
        self.assertIn("checkpoint_hash", data)
        self.assertIn("calibration_status", data)
        self.assertTrue(data["is_suitable_for_evaluation"])

    def test_api_status_includes_health_and_warning(self):
        """Validates GET /api/status includes health_check object and warning status."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("health_check", data)
        self.assertIn("warning", data)
        self.assertIn("model", data)
        self.assertIn("calibration_status", data["model"])
        self.assertIn("is_suitable_for_evaluation", data["model"])

    def test_uninitialized_model_reports_not_ready(self):
        """Validates uninitialized/missing checkpoint model marks not ready."""
        cfg = ModelConfig()
        cfg.checkpoint_path = "non_existent_path.pth"
        classifier = VideoActionClassifier(cfg)
        health = classifier.get_health_report()

        self.assertFalse(classifier.is_ready)
        self.assertFalse(health["is_suitable_for_evaluation"])
        self.assertEqual(health["calibration_status"], "UNCALIBRATED")


if __name__ == "__main__":
    unittest.main()
