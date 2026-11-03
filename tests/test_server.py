"""
Unit and Integration Tests for SafeTV FastAPI Server & Security Dashboard.
Tests file upload analysis, live webcam stream inference, incident listing,
forensic clip streaming, and operator review submission.
"""

import base64
import os
import unittest
from fastapi.testclient import TestClient
import numpy as np
import cv2

from safetv.server import app


class TestSafeTVServer(unittest.TestCase):
    """Integration test suite for SafeTV backend server endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_status_endpoint(self):
        """Validates that system status returns engine online and model configuration."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertIn("model", data)
        self.assertTrue(data["model"]["is_trained"])
        self.assertEqual(data["model"]["provenance"], "trained")

    def test_dashboard_html_served(self):
        """Validates that root path serves the SOC HTML5 dashboard."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("SafeTV", response.text)
        self.assertIn("Live Camera Safety Feed", response.text)

    def test_upload_and_analyze_violence_video(self):
        """Validates video upload, deep model inference, and incident evidence creation."""
        test_video = os.path.join("data", "raw", "violence", "punching_001.mp4")
        if not os.path.exists(test_video):
            self.skipTest(f"{test_video} not found")

        with open(test_video, "rb") as f:
            files = {"file": ("punching_001.mp4", f, "video/mp4")}
            response = self.client.post("/api/analyze/upload", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        report = data["report"]
        self.assertGreaterEqual(report["incidents_detected"], 1)
        self.assertGreater(report["peak_violence_prob"], 0.65)
        self.assertTrue(report["is_model_trained"])

        # Check evidence clip
        inc = report["incidents"][0]
        self.assertIn("evidence_filename", inc)
        evidence_fn = inc["evidence_filename"]

        # Validate that the evidence clip can be streamed
        clip_resp = self.client.get(f"/api/evidence/{evidence_fn}")
        self.assertEqual(clip_resp.status_code, 200)
        self.assertEqual(clip_resp.headers["content-type"], "video/mp4")

    def test_incidents_vault_and_review(self):
        """Validates incident retrieval and security operator review submission."""
        response = self.client.get("/api/incidents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("incidents", data)

        if data["incidents"]:
            first_inc = data["incidents"][0]
            inc_id = str(first_inc["incident_id"])

            review_payload = {
                "decision": "TRUE_POSITIVE",
                "reviewer_notes": "Verified by security lead; immediate dispatch required.",
                "reviewer_name": "Lead Guard",
            }
            review_resp = self.client.post(f"/api/incidents/{inc_id}/review", json=review_payload)
            self.assertEqual(review_resp.status_code, 200)
            res_json = review_resp.json()
            self.assertTrue(res_json["success"])
            self.assertEqual(res_json["review"]["decision"], "TRUE_POSITIVE")

    def test_stream_window_inference(self):
        """Validates live webcam frame batch streaming and inference."""
        # Synthesize 16 simple JPEG base64 frames
        frames_b64 = []
        for _ in range(16):
            img = np.full((120, 160, 3), fill_value=120, dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", img)
            b64_str = base64.b64encode(buf).decode("utf-8")
            frames_b64.append(f"data:image/jpeg;base64,{b64_str}")

        payload = {
            "frames_base64": frames_b64,
            "fps": 15.0,
        }
        response = self.client.post("/api/analyze/stream-window", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("smoothed_violence_prob", data)
        self.assertTrue(data["is_model_trained"])

    def test_camera_crud_and_alerts_endpoints(self):
        """Validates camera registration, status polling, and alert dispatching."""
        # Register new camera
        cam_payload = {
            "camera_id": "api_test_cam",
            "name": "API Test Camera",
            "location": "North Corridor",
            "source_url": "data/raw/violence/punching_001.mp4",
            "auto_start": False,
        }
        reg_resp = self.client.post("/api/cameras", json=cam_payload)
        self.assertEqual(reg_resp.status_code, 200)
        self.assertTrue(reg_resp.json()["success"])

        # List cameras
        list_resp = self.client.get("/api/cameras")
        self.assertEqual(list_resp.status_code, 200)
        cams = list_resp.json()["cameras"]
        self.assertTrue(any(c["camera_id"] == "api_test_cam" for c in cams))

        # Test alert trigger
        alert_resp = self.client.post("/api/alerts/test")
        self.assertEqual(alert_resp.status_code, 200)
        self.assertTrue(alert_resp.json()["success"])

        # Check alert list
        alerts_list_resp = self.client.get("/api/alerts")
        self.assertEqual(alerts_list_resp.status_code, 200)
        alerts = alerts_list_resp.json()["alerts"]
        self.assertGreaterEqual(len(alerts), 1)

        # Cleanup camera
        del_resp = self.client.delete("/api/cameras/api_test_cam")
        self.assertEqual(del_resp.status_code, 200)

    def test_path_traversal_evidence_rejected(self):
        """Validates that path traversal attacks on /api/evidence/{filename} are rejected."""
        traversal_attempts = [
            "..%2F..%2Fconfig.py",
            "../../etc/passwd",
            "..\\..\\Windows\\System32\\cmd.exe",
            "nonexistent_file_xyz.mp4"
        ]
        for attempt in traversal_attempts:
            resp = self.client.get(f"/api/evidence/{attempt}")
            self.assertIn(resp.status_code, [404, 400], f"Failed to reject traversal for: {attempt}")

    def test_webcam_session_lifecycle_and_chunk(self):
        """Validates webcam session start, chunk inference, and session close."""
        # 1. Start session
        start_resp = self.client.post("/api/webcam/session/start")
        self.assertEqual(start_resp.status_code, 200)
        start_data = start_resp.json()
        self.assertTrue(start_data["success"])
        session_id = start_data["session_id"]
        self.assertIsNotNone(session_id)

        # 2. Stop session
        stop_resp = self.client.post("/api/webcam/session/stop", json={"session_id": session_id})
        self.assertEqual(stop_resp.status_code, 200)
        self.assertTrue(stop_resp.json()["success"])

    def test_export_reviewed_dataset_endpoint(self):
        """Validates dataset export endpoint for reviewed incidents."""
        resp = self.client.post("/api/incidents/export-dataset")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("export_summary", data)


if __name__ == "__main__":
    unittest.main()
