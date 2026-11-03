"""
SafeTV Persistent Webcam Session Management Module.
Maintains stateful sliding window extraction, probability smoothing,
and hysteresis incident detection across sequential browser HTTP batches.
Prevents state-wipe between chunked frames and manages session lifecycles.
"""

import collections
import time
import threading
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from safetv.config import SafeTVConfig
from safetv.evidence import EvidenceExtractor
from safetv.incident_classifier import Incident, IncidentClassifier
from safetv.model import ModelNotReadyError, VideoActionClassifier
from safetv.temporal_smoothing import TemporalSmoother
from safetv.video_loader import VideoWindowExtractor


class WebcamSession:
    """Stateful persistent monitoring session for a live browser webcam feed."""

    def __init__(self, session_id: str, config: Optional[SafeTVConfig] = None):
        self.session_id = session_id
        self.config = config or SafeTVConfig()

        self.extractor = VideoWindowExtractor(self.config.extraction)
        self.model = VideoActionClassifier(self.config.model)
        self.smoother = TemporalSmoother(self.config.smoothing)
        self.incident_classifier = IncidentClassifier(self.config.incident)
        self.evidence_extractor = EvidenceExtractor(self.config.evidence)

        self.lock = threading.Lock()
        self.frame_buffer: collections.deque = collections.deque(maxlen=150)
        self.total_frames_processed: int = 0
        self.last_activity: float = time.time()
        self.accumulated_window_frames: List[np.ndarray] = []
        self.last_prob: float = 0.0
        self.peak_prob: float = 0.0

    def add_frames_and_infer(
        self,
        new_frames: List[np.ndarray],
        fps: float = 15.0
    ) -> Tuple[float, bool, List[Incident]]:
        """
        Appends new frames to persistent rolling window buffer, runs inference
        on newly completed temporal windows, and updates incident state machine.
        """
        if not self.model.is_ready:
            raise ModelNotReadyError(
                f"Webcam inference disabled: {self.model.model_status_message}. "
                "A valid trained checkpoint must be loaded."
            )

        with self.lock:
            self.last_activity = time.time()
            completed_incidents: List[Incident] = []

            for f in new_frames:
                self.frame_buffer.append(f)
                self.accumulated_window_frames.append(f)
                self.total_frames_processed += 1

                # Check if a complete sliding window span is ready
                req_span = (self.config.extraction.window_size - 1) * self.config.extraction.temporal_stride + 1
                if len(self.accumulated_window_frames) >= req_span:
                    # Subsample frames
                    stride = self.config.extraction.temporal_stride
                    size = self.config.extraction.window_size
                    sampled = [self.accumulated_window_frames[i * stride] for i in range(size)]

                    tensor_4d = self.extractor._resize_and_normalize(sampled)

                    preds = self.model.predict_window(tensor_4d, allow_uncalibrated=False)
                    raw_prob = float(preds["violence_prob"])

                    smoothed_prob = self.smoother.update(raw_prob)
                    self.last_prob = round(smoothed_prob, 4)
                    if smoothed_prob > self.peak_prob:
                        self.peak_prob = round(smoothed_prob, 4)

                    video_sec = self.total_frames_processed / fps

                    # Update persistent incident classifier (state is NOT wiped)
                    if self.model.is_ready:
                        inc = self.incident_classifier.process_window(
                            smoothed_prob=smoothed_prob,
                            timestamp_sec=video_sec,
                            frame_idx=self.total_frames_processed
                        )
                        if inc is not None:
                            # Slice evidence from ring buffer
                            self._slice_evidence_clip(inc, fps)
                            completed_incidents.append(inc)

                    # Advance window by window_step
                    step = max(1, self.config.extraction.window_step)
                    self.accumulated_window_frames = self.accumulated_window_frames[step:]

            is_active = self.incident_classifier.is_incident_active if self.model.is_ready else False
            return self.last_prob, is_active, completed_incidents

    def _slice_evidence_clip(self, incident: Incident, fps: float) -> None:
        """Extract evidence clip from the webcam session buffer."""
        frames = list(self.frame_buffer)
        if frames and self.config.evidence.save_evidence_clips:
            self.evidence_extractor.extract_clip_from_frames(
                frames=frames,
                fps=fps,
                incident=incident,
                source_name=f"webcam_{self.session_id[:8]}",
                checkpoint_hash=self.model.checkpoint_hash,
            )

    def close(self) -> List[Incident]:
        """Finalize any open incidents and clean up session."""
        with self.lock:
            final_incidents = []
            if self.model.is_ready:
                final = self.incident_classifier.finalize()
                for inc in final:
                    if not inc.evidence_path:
                        self._slice_evidence_clip(inc, fps=15.0)
                    final_incidents.append(inc)
            self.frame_buffer.clear()
            self.accumulated_window_frames.clear()
            return final_incidents


class WebcamSessionManager:
    """Manages active browser webcam monitoring sessions and automatic cleanup."""

    def __init__(self, config: Optional[SafeTVConfig] = None):
        self.config = config or SafeTVConfig()
        self._sessions: Dict[str, WebcamSession] = {}
        self._lock = threading.Lock()

    def create_session(self) -> str:
        """Initialize a new persistent webcam monitoring session."""
        session_id = f"wcam_{uuid.uuid4().hex[:12]}"
        session = WebcamSession(session_id, self.config)
        with self._lock:
            self._cleanup_stale_sessions()
            self._sessions[session_id] = session
        return session_id

    def get_or_create_session(self, session_id: Optional[str] = None) -> WebcamSession:
        """Retrieve existing session or create a new one."""
        with self._lock:
            self._cleanup_stale_sessions()
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]

            sid = session_id or f"wcam_{uuid.uuid4().hex[:12]}"
            session = WebcamSession(sid, self.config)
            self._sessions[sid] = session
            return session

    def close_session(self, session_id: str) -> List[Incident]:
        """Terminate and finalize a webcam session."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            return session.close()
        return []

    def _cleanup_stale_sessions(self, max_idle_sec: float = 300.0) -> None:
        """Remove idle sessions that have not received frames in 5 minutes."""
        now = time.time()
        stale_ids = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_activity) > max_idle_sec
        ]
        for sid in stale_ids:
            s = self._sessions.pop(sid, None)
            if s:
                s.close()
