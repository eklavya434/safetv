"""
SafeTV CCTV & Multi-Camera Ingestion Engine.
Manages persistent threaded workers for RTSP / HTTP surveillance camera feeds,
rolling spatiotemporal frame buffers, automated reconnection, and real-time alert dispatching.
Enforces a consistent timestamp model (camera-relative seconds + wall-clock capture).
"""

import collections
import datetime
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from safetv.alerts import AlertDispatcher
from safetv.config import SafeTVConfig
from safetv.evidence import EvidenceExtractor
from safetv.incident_classifier import Incident, IncidentClassifier
from safetv.model import ModelNotReadyError, VideoActionClassifier
from safetv.temporal_smoothing import TemporalSmoother
from safetv.video_loader import VideoWindowExtractor

logger = logging.getLogger("safetv.cctv")


@dataclass
class BufferedFrame:
    """Frame structure with synchronized timestamp metadata."""
    frame_index: int
    video_sec: float              # Camera-relative seconds since session start
    monotonic_sec: float          # Local monotonic clock
    wall_clock_iso: str           # ISO 8601 wall-clock capture time
    rgb_frame: np.ndarray


@dataclass
class CameraConfig:
    """Surveillance camera stream metadata."""
    camera_id: str
    name: str
    location: str
    source_url: str                     # RTSP URL (e.g. rtsp://192.168.1.50:554/live) or local video path
    enabled: bool = True
    target_fps: float = 15.0
    buffer_seconds: float = 8.0         # Duration of rolling in-memory frame ring buffer (>= pre + post buffers)
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

        # Rolling frame ring buffer: stores BufferedFrame
        buffer_len = max(90, int(self.camera.target_fps * self.camera.buffer_seconds))
        self._frame_buffer: collections.deque[BufferedFrame] = collections.deque(maxlen=buffer_len)

        # Pending incident post-buffer queue
        self._pending_evidence_incidents: List[Tuple[Incident, float]] = []

        # Telemetry
        self.status = "STOPPED"  # STOPPED, CONNECTING, RUNNING, RECONNECTING, ERROR
        self.fps_actual = 0.0
        self.last_prob = 0.0
        self.peak_prob = 0.0
        self.incidents_detected_count = 0
        self.last_error: Optional[str] = None
        self.last_frame_time: Optional[float] = None
        self.session_start_time: Optional[float] = None
        self.session_frame_count: int = 0

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
                self.session_start_time = time.monotonic()
                self.session_frame_count = 0
                self.incident_classifier.reset()
                self.smoother.reset()

                fps_stream = cap.get(cv2.CAP_PROP_FPS)
                target_fps = fps_stream if (fps_stream and fps_stream > 0 and not np.isnan(fps_stream)) else self.camera.target_fps

                start_clock = time.time()
                sec_counter = 0

                window_frames: List[np.ndarray] = []

                while not self._stop_event.is_set():
                    ret, bgr = cap.read()
                    if not ret:
                        break

                    now_mono = time.monotonic()
                    now_wall = datetime.datetime.now().isoformat()
                    video_sec = self.session_frame_count / target_fps

                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    b_frame = BufferedFrame(
                        frame_index=self.session_frame_count,
                        video_sec=video_sec,
                        monotonic_sec=now_mono,
                        wall_clock_iso=now_wall,
                        rgb_frame=rgb,
                    )

                    with self._lock:
                        self._frame_buffer.append(b_frame)
                        self.last_frame_time = time.time()

                    window_frames.append(rgb)
                    self.session_frame_count += 1
                    sec_counter += 1

                    # Compute rolling actual FPS
                    elapsed = time.time() - start_clock
                    if elapsed >= 1.0:
                        self.fps_actual = round(sec_counter / elapsed, 1)
                        sec_counter = 0
                        start_clock = time.time()

                    # Process window every window_step frames
                    required_span = (self.config.extraction.window_size - 1) * self.config.extraction.temporal_stride + 1
                    if len(window_frames) >= required_span:
                        self._process_accumulated_window(window_frames, video_sec, self.session_frame_count)
                        step = max(1, self.config.extraction.window_step)
                        window_frames = window_frames[step:]

                    # Check pending evidence clips for post-buffer readiness
                    self._check_pending_evidence(video_sec, target_fps)

                    time.sleep(max(0.001, 1.0 / target_fps - 0.005))

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

    def _process_accumulated_window(self, frames_list: List[np.ndarray], current_video_sec: float, current_frame_idx: int) -> None:
        """Extract spatiotemporal window tensor, perform AI inference, and update incident tracker."""
        if not self.model.is_ready:
            return

        try:
            stride = self.config.extraction.temporal_stride
            size = self.config.extraction.window_size
            sampled = [frames_list[i * stride] for i in range(size)]

            tensor_4d = self.extractor._resize_and_normalize(sampled)

            preds = self.model.predict_window(tensor_4d)
            raw_prob = float(preds["violence_prob"])
            smoothed_prob = self.smoother.update(raw_prob)

            with self._lock:
                self.last_prob = round(smoothed_prob, 4)
                if smoothed_prob > self.peak_prob:
                    self.peak_prob = round(smoothed_prob, 4)

            # Pass video-relative timestamp to maintain unified timeline
            completed_incident = self.incident_classifier.process_window(
                smoothed_prob=smoothed_prob,
                timestamp_sec=current_video_sec,
                frame_idx=current_frame_idx
            )

            if completed_incident is not None:
                self.incidents_detected_count += 1
                logger.warning(
                    f" [!] INCIDENT #{completed_incident.incident_id} DETECTED on camera '{self.camera.name}'! "
                    f"({completed_incident.start_time_sec:.2f}s - {completed_incident.end_time_sec:.2f}s, "
                    f"Peak: {completed_incident.peak_probability * 100:.1f}%)"
                )
                # Queue incident for evidence extraction once post-buffer has arrived
                post_target_sec = completed_incident.end_time_sec + self.config.evidence.post_buffer_sec
                with self._lock:
                    self._pending_evidence_incidents.append((completed_incident, post_target_sec))

        except Exception as e:
            logger.error(f"Window processing error on camera {self.camera.camera_id}: {e}")

    def _check_pending_evidence(self, current_video_sec: float, fps: float) -> None:
        """Check if post-buffer is available for pending incidents, then slice clip and alert."""
        ready_incidents = []
        with self._lock:
            remaining = []
            for inc, target_sec in self._pending_evidence_incidents:
                if current_video_sec >= target_sec:
                    ready_incidents.append(inc)
                else:
                    remaining.append((inc, target_sec))
            self._pending_evidence_incidents = remaining

        for inc in ready_incidents:
            self._slice_and_dispatch_incident(inc, fps)

    def _slice_and_dispatch_incident(self, incident: Incident, fps: float) -> None:
        """Extract buffered evidence frames matching camera-relative interval."""
        pre_buf = self.config.evidence.pre_buffer_sec
        post_buf = self.config.evidence.post_buffer_sec
        t_start = max(0.0, incident.start_time_sec - pre_buf)
        t_end = incident.end_time_sec + post_buf

        with self._lock:
            matched_frames = [
                f.rgb_frame for f in self._frame_buffer
                if t_start <= f.video_sec <= t_end
            ]

        if matched_frames and self.config.evidence.save_evidence_clips:
            clip_path = self.evidence_extractor.extract_clip_from_frames(
                frames=matched_frames,
                fps=fps,
                incident=incident,
                source_name=self.camera.name
            )
            incident.evidence_path = clip_path

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
