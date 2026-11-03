"""
Unit Tests for SafeTV Alert Dispatch Engine.
Validates alert rate-limiting, deduplication, payload formatting, and history logging.
"""

import os
import shutil
import tempfile
import unittest

from safetv.alerts import AlertConfig, AlertDispatcher
from safetv.incident_classifier import Incident


class TestAlertDispatcher(unittest.TestCase):
    """Unit tests for alert generation, rate-limiting, and webhook dispatching."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.alert_log = os.path.join(self.test_dir, "test_alerts.json")
        self.config = AlertConfig(
            cooldown_seconds=1.0,
            alert_log_file=self.alert_log,
            enable_webhook=False,
            include_evidence_link=True,
            base_server_url="http://localhost:8000"
        )
        self.dispatcher = AlertDispatcher(self.config)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_alert_dispatch_and_formatting(self):
        """Validates that a detected incident dispatches a correctly formatted alert."""
        inc = Incident(
            incident_id=1,
            start_time_sec=2.0,
            end_time_sec=5.0,
            duration_sec=3.0,
            start_frame=60,
            end_frame=150,
            peak_probability=0.89,
            mean_probability=0.82,
            severity="CRITICAL",
            evidence_path="evidence/INC-0001_clip.mp4",
        )

        alert = self.dispatcher.dispatch_incident_alert(
            incident=inc,
            camera_id="cam_gate_01",
            camera_name="Gate 1"
        )

        self.assertIsNotNone(alert)
        self.assertEqual(alert.camera_id, "cam_gate_01")
        self.assertEqual(alert.severity, "CRITICAL")
        self.assertEqual(alert.status, "DISPATCHED")
        self.assertIn("Gate 1", alert.message)
        self.assertIn("INC-0001_clip.mp4", alert.evidence_clip_url)

        # Check persistence
        recent = self.dispatcher.get_recent_alerts()
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["alert_id"], alert.alert_id)

    def test_alert_rate_limiting_cooldown(self):
        """Validates that rapid consecutive alerts for the same camera are suppressed."""
        inc = Incident(
            incident_id=2,
            start_time_sec=1.0,
            end_time_sec=2.0,
            duration_sec=1.0,
            start_frame=30,
            end_frame=60,
            peak_probability=0.75,
            mean_probability=0.72,
            severity="HIGH",
        )

        # First alert passes
        a1 = self.dispatcher.dispatch_incident_alert(inc, camera_id="cam_lobby")
        self.assertIsNotNone(a1)

        # Immediate second alert for same camera is suppressed by cooldown
        a2 = self.dispatcher.dispatch_incident_alert(inc, camera_id="cam_lobby")
        self.assertIsNone(a2)

        # Alert for different camera passes immediately
        a3 = self.dispatcher.dispatch_incident_alert(inc, camera_id="cam_perimeter")
        self.assertIsNotNone(a3)


if __name__ == "__main__":
    unittest.main()
