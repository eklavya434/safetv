"""
Unit and Integration Tests for SafeTV CCTV Multi-Camera Engine.
Validates camera registration, lifecycle management, and stream workers.
"""

import os
import shutil
import tempfile
import time
import unittest

from safetv.alerts import AlertConfig, AlertDispatcher
from safetv.cctv import BufferedFrame, CameraConfig, CameraManager, RTSPStreamWorker
from safetv.incident_classifier import Incident


class TestCCTVEngine(unittest.TestCase):
    """Unit tests for camera registry, configuration, and worker lifecycle."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.reg_file = os.path.join(self.test_dir, "test_cameras.json")
        self.alert_log = os.path.join(self.test_dir, "alerts.json")
        self.dispatcher = AlertDispatcher(AlertConfig(alert_log_file=self.alert_log, enable_webhook=False))
        self.manager = CameraManager(
            registry_file=self.reg_file,
            alert_dispatcher=self.dispatcher
        )

    def tearDown(self):
        self.manager.stop_all()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_camera_registration_and_persistence(self):
        """Validates adding and removing cameras from the registry."""
        cam = self.manager.register_camera(
            camera_id="cam_east_01",
            name="East Entrance",
            location="East Gate",
            source_url="data/raw/violence/punching_001.mp4",
        )
        self.assertEqual(cam.camera_id, "cam_east_01")
        self.assertIn("cam_east_01", self.manager.cameras)

        statuses = self.manager.get_all_camera_statuses()
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["camera_id"], "cam_east_01")
        self.assertEqual(statuses[0]["status"], "STOPPED")

        # Delete camera
        deleted = self.manager.delete_camera("cam_east_01")
        self.assertTrue(deleted)
        self.assertEqual(len(self.manager.cameras), 0)

    def test_camera_worker_start_stop(self):
        """Validates running background stream worker on a test video file."""
        video_path = os.path.join("data", "raw", "violence", "punching_001.mp4")
        if not os.path.exists(video_path):
            self.skipTest(f"{video_path} not found")

        cam = self.manager.register_camera(
            camera_id="cam_test_worker",
            name="Test Worker Cam",
            location="Lab",
            source_url=video_path,
        )

        started = self.manager.start_camera("cam_test_worker")
        self.assertTrue(started)

        # Allow worker thread to initialize
        time.sleep(1.0)
        status = self.manager.workers["cam_test_worker"].get_status()
        self.assertIn(status["status"], ["RUNNING", "CONNECTING", "STOPPED"])

        stopped = self.manager.stop_camera("cam_test_worker")
        self.assertTrue(stopped)
        self.assertEqual(self.manager.workers["cam_test_worker"].status, "STOPPED")

    def test_buffered_frame_timestamp_consistency(self):
        """Validates that BufferedFrame holds video-relative and monotonic timestamps accurately."""
        dummy_rgb = (100 * (None or 1)).to_bytes(0, 'big') if False else None
        b_frame = BufferedFrame(
            frame_index=150,
            video_sec=5.0, # 150 frames @ 30 FPS = 5.0 seconds
            monotonic_sec=1024.5,
            wall_clock_iso="2026-11-03T11:00:00Z",
            rgb_frame=dummy_rgb,
        )
        self.assertEqual(b_frame.frame_index, 150)
        self.assertEqual(b_frame.video_sec, 5.0)
        self.assertEqual(b_frame.monotonic_sec, 1024.5)
        self.assertEqual(b_frame.wall_clock_iso, "2026-11-03T11:00:00Z")


if __name__ == "__main__":
    unittest.main()
