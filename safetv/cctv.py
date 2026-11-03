"""
SafeTV CCTV & Multi-Camera Ingestion Engine.
Manages persistent threaded workers for RTSP / HTTP surveillance camera feeds,
rolling spatiotemporal frame buffers, automated reconnection, and real-time alert dispatching.
"""

import collections
import datetime
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from safetv.alerts import AlertDispatcher
from safetv.config import SafeTVConfig
from safetv.evidence import EvidenceExtractor
from safetv.incident_classifier import Incident, IncidentClassifier
from safetv.model import VideoActionClassifier
from safetv.temporal_smoothing import TemporalSmoother
from safetv.video_loader import VideoWindowExtractor

logger = logging.getLogger("safetv.cctv")


@dataclass
class CameraConfig:
    """Surveillance camera stream metadata."""
    camera_id: str
    name: str
    location: str
    source_url: str                     # RTSP URL (e.g. rtsp://192.168.1.50:554/live) or local video path
    enabled: bool = True
    target_fps: float = 15.0
    buffer_seconds: float = 6.0         # Duration of rolling in-memory frame ring buffer
    reconnect_interval_sec: float = 3.0 # Interval before retrying a dropped connection

    def to_dict(self) -> dict:
        return asdict(self)


class RTSPStreamWorker:
    """
    Dedicated background worker thread for an active RTSP camera feed.
    Captures frames, maintains a rolling FIFO ring buffer, executes windowed inference,
    and dispatches safety alerts upon incident detection.
    """

    def __init__(
        self,
        camera: CameraConfig,
        pipeline_config: Optional[SafeTVConfig] = None,
        alert_dispatcher: Optional[AlertDispatcher] = None
    ):
        self.camera = camera
        self.config = pipeline_config or SafeTVConfig()
        self.alert_dispatcher = alert_dispatcher

        # Inference components dedicated to this stream
        self.extractor = VideoWindowExtractor(self.config.extraction)
        self.model = VideoActionClassifier(self.config.model)
        self.smoother = TemporalSmoother(self.config.smoothing)
        self.incident_classifier = IncidentClassifier(self.config.incident)
        self.evidence_extractor = EvidenceExtractor(self.config.evidence)

        # Thread management & state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Rolling frame ring buffer: stores (timestamp, rgb_frame)
        buffer_len = max(60, int(self.camera.target_fps * self.camera.buffer_seconds))
        self._frame_buffer: collections.deque = collections.deque(maxlen=buffer_len)

        # Telemetry
        self.status = "STOPPED"  # STOPPED, CONNECTING, RUNNING, RECONNECTING, ERROR
        self.fps_actual = 0.0
        self.last_prob = 0.0
        self.peak_prob = 0.0
        self.incidents_detected_count = 0
        self.last_error: Optional[str] = None
        self.last_frame_time: Optional[float] = None

    def start(self) -> None:
        """Start the camera ingestion and AI monitoring thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, name=f"CamWorker-{self.camera.camera_id}", daemon=True)
            self._thread.start()
            self.status = "CONNECTING"
            logger.info(f"Started worker for camera '{self.camera.name}' ({self.camera.camera_id})")

    def stop(self) -> None:
        """Stop worker and release camera stream resources."""
        with self._lock:
            self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.status = "STOPPED"
        logger.info(f"Stopped worker for camera '{self.camera.name}'")

    def _run_loop(self) -> None:
        """Main camera capture and inference loop with automatic reconnection."""
        while not self._stop_event.is_set():
            cap = None
            try:
                self.status = "CONNECTING"
                cap = cv2.VideoCapture(self.camera.source_url)
                if not cap.isOpened():
                    raise IOError(f"Could not connect to video source: {self.camera.source_url}")

                self.status = "RUNNING"
                self.last_error = None
                frame_count = 0
                start_clock = time.time()
                last_window_time = time.time()

                # Inference window accumulation buffer
                window_frames = []

                while not self._stop_event.is_set():
                    ret, bgr = cap.read()
                    if not ret:
                        # Video ended (if file) or network dropped (if RTSP)
                        break

                    now = time.time()
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    with self._lock:
                        self._frame_buffer.append((now, rgb))
                        self.last_frame_time = now

                    window_frames.append(rgb)
                    frame_count += 1

                    # Compute rolling actual FPS
                    elapsed = now - start_clock
                    if elapsed >= 1.0:
                        self.fps_actual = round(frame_count / elapsed, 1)
                        frame_count = 0
                        start_clock = now

                    # Process window every window_step frames
                    required_span = (self.config.extraction.window_size - 1) * self.config.extraction.temporal_stride + 1
                    if len(window_frames) >= required_span:
                        self._process_accumulated_window(window_frames)
                        # Slide forward by window_step
                        step = max(1, self.config.extraction.window_step)
                        window_frames = window_frames[step:]

                    # Throttle to target FPS
                    time.sleep(max(0.001, 1.0 / self.camera.target_fps - 0.005))

            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Camera '{self.camera.camera_id}' error: {e}")
            finally:
                if cap is not None:
                    cap.release()

            if not self._stop_event.is_set():
                self.status = "RECONNECTING"
                time.sleep(self.camera.reconnect_interval_sec)

        self.status = "STOPPED"

    def _process_accumulated_window(self, frames_list: List[np.ndarray]) -> None:
        """Extract spatiotemporal window tensor, perform AI inference, and update incident tracker."""
        try:
            # Subsample frames according to temporal stride
            stride = self.config.extraction.temporal_stride
            size = self.config.extraction.window_size
            sampled = [frames_list[i * stride] for i in range(size)]

            # Convert to normalized 4D tensor
            tensor_4d = self.extractor._resize_and_normalize(sampled)

            # Deep AI inference
            preds = self.model.predict_window(tensor_4d)
            raw_prob = float(preds["violence_prob"])
            smoothed_prob = self.smoother.update(raw_prob)

            with self._lock:
                self.last_prob = round(smoothed_prob, 4)
                if smoothed_prob > self.peak_prob:
                    self.peak_prob = round(smoothed_prob, 4)

            now_sec = time.time()
            completed_incident = self.incident_classifier.process_window(
                smoothed_prob=smoothed_prob,
                timestamp_sec=now_sec,
                frame_idx=0
            )

            if completed_incident is not None:
                self._handle_completed_incident(completed_incident)

        except Exception as e:
            logger.error(f"Window processing error on camera {self.camera.camera_id}: {e}")

    def _handle_completed_incident(self, incident: Incident) -> None:
        """Slice evidence video from ring buffer and dispatch notification."""
        self.incidents_detected_count += 1
        logger.warning(f" [!] INCIDENT #{incident.incident_id} DETECTED on camera '{self.camera.name}'! (Peak: {incident.peak_probability * 100:.1f}%)")

        # Snapshot buffered frames for evidence clip
        with self._lock:
            buffered_snapshots = [f[1] for f in self._frame_buffer]

        if buffered_snapshots and self.config.evidence.save_evidence_clips:
            # Slices MP4 clip and metadata JSON
            clip_path = self.evidence_extractor.extract_clip_from_frames(
                frames=buffered_snapshots,
                fps=self.fps_actual if self.fps_actual > 0 else self.camera.target_fps,
                incident=incident,
                source_name=self.camera.name
            )
            incident.evidence_path = clip_path

        # Dispatch real-time alert via Webhook / Email / Audit log
        if self.alert_dispatcher:
            self.alert_dispatcher.dispatch_incident_alert(
                incident=incident,
                camera_id=self.camera.camera_id,
                camera_name=self.camera.name
            )

    def get_status(self) -> dict:
        """Return live telemetry snapshot."""
        with self._lock:
            return {
                "camera_id": self.camera.camera_id,
                "name": self.camera.name,
                "location": self.camera.location,
                "source_url": self.camera.source_url,
                "status": self.status,
                "fps": self.fps_actual,
                "current_violence_prob": self.last_prob,
                "peak_violence_prob": self.peak_prob,
                "incidents_detected": self.incidents_detected_count,
                "is_incident_active": self.incident_classifier.is_incident_active,
                "last_error": self.last_error,
                "last_frame_time": self.last_frame_time,
            }


class CameraManager:
    """
    Centralized multi-camera registry managing stream lifecycle, configuration, and monitoring.
    """

    def __init__(
        self,
        registry_file: str = "cameras.json",
        pipeline_config: Optional[SafeTVConfig] = None,
        alert_dispatcher: Optional[AlertDispatcher] = None
    ):
        self.registry_file = registry_file
        self.pipeline_config = pipeline_config or SafeTVConfig()
        self.alert_dispatcher = alert_dispatcher or AlertDispatcher()
        self._lock = threading.Lock()
        self.workers: Dict[str, RTSPStreamWorker] = {}
        self.cameras: Dict[str, CameraConfig] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load cameras from persistence file."""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        cam = CameraConfig(**item)
                        self.cameras[cam.camera_id] = cam
            except Exception as e:
                logger.error(f"Failed to load camera registry: {e}")

    def _save_registry(self) -> None:
        """Save camera list to disk."""
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in self.cameras.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save camera registry: {e}")

    def register_camera(
        self,
        camera_id: str,
        name: str,
        location: str,
        source_url: str,
        auto_start: bool = False
    ) -> CameraConfig:
        """Register a new surveillance camera stream."""
        with self._lock:
            cam = CameraConfig(
                camera_id=camera_id,
                name=name,
                location=location,
                source_url=source_url,
            )
            self.cameras[camera_id] = cam
            self._save_registry()

        if auto_start:
            self.start_camera(camera_id)
        return cam

    def delete_camera(self, camera_id: str) -> bool:
        """Stop worker and remove camera from registry."""
        self.stop_camera(camera_id)
        with self._lock:
            if camera_id in self.cameras:
                del self.cameras[camera_id]
                self._save_registry()
                return True
            return False

    def start_camera(self, camera_id: str) -> bool:
        """Spin up background RTSP monitoring worker for a camera."""
        with self._lock:
            cam = self.cameras.get(camera_id)
            if not cam:
                return False

            if camera_id not in self.workers:
                worker = RTSPStreamWorker(
                    camera=cam,
                    pipeline_config=self.pipeline_config,
                    alert_dispatcher=self.alert_dispatcher
                )
                self.workers[camera_id] = worker
            else:
                worker = self.workers[camera_id]

        worker.start()
        return True

    def stop_camera(self, camera_id: str) -> bool:
        """Stop worker for a camera."""
        with self._lock:
            worker = self.workers.get(camera_id)
        if worker:
            worker.stop()
            return True
        return False

    def stop_all(self) -> None:
        """Stop all active camera workers."""
        with self._lock:
            workers_list = list(self.workers.values())
        for w in workers_list:
            w.stop()

    def get_all_camera_statuses(self) -> List[dict]:
        """Collect statuses of all registered cameras."""
        with self._lock:
            statuses = []
            for cam_id, cam in self.cameras.items():
                worker = self.workers.get(cam_id)
                if worker:
                    statuses.append(worker.get_status())
                else:
                    statuses.append({
                        "camera_id": cam.camera_id,
                        "name": cam.name,
                        "location": cam.location,
                        "source_url": cam.source_url,
                        "status": "STOPPED",
                        "fps": 0.0,
                        "current_violence_prob": 0.0,
                        "peak_violence_prob": 0.0,
                        "incidents_detected": 0,
                        "is_incident_active": False,
                        "last_error": None,
                        "last_frame_time": None,
                    })
            return statuses
