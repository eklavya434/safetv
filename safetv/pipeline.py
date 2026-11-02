"""
SafeTV Violence Detection Pipeline.
Unifies spatiotemporal window extraction, model inference, temporal smoothing,
and incident state classification into a cohesive end-to-end processing pipeline.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Union

import numpy as np

from .config import SafeTVConfig
from .incident_classifier import Incident, IncidentClassifier
from .model import VideoActionClassifier
from .temporal_smoothing import TemporalSmoother
from .video_loader import VideoWindow, VideoWindowExtractor


@dataclass
class WindowResult:
    """Detailed inference result for a single temporal window."""
    window_idx: int
    start_time_sec: float
    end_time_sec: float
    center_time_sec: float
    start_frame: int
    end_frame: int
    raw_violence_prob: float
    smoothed_violence_prob: float
    predicted_label: str
    incident_active: bool

    def to_dict(self) -> dict:
        return {
            "window_idx": self.window_idx,
            "start_time_sec": round(self.start_time_sec, 2),
            "end_time_sec": round(self.end_time_sec, 2),
            "center_time_sec": round(self.center_time_sec, 2),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "raw_violence_prob": round(self.raw_violence_prob, 4),
            "smoothed_violence_prob": round(self.smoothed_violence_prob, 4),
            "predicted_label": self.predicted_label,
            "incident_active": self.incident_active,
        }


@dataclass
class PipelineReport:
    """Comprehensive analysis report for an analyzed video stream or file."""
    video_source: str
    total_duration_sec: float
    total_windows_processed: int
    incidents_detected: int
    incidents: List[Dict]
    mean_violence_prob: float
    peak_violence_prob: float
    fps: float
    execution_time_sec: float
    window_timeline: List[Dict]

    def to_dict(self) -> dict:
        return {
            "video_source": self.video_source,
            "total_duration_sec": round(self.total_duration_sec, 2),
            "total_windows_processed": self.total_windows_processed,
            "incidents_detected": self.incidents_detected,
            "incidents": self.incidents,
            "mean_violence_prob": round(self.mean_violence_prob, 4),
            "peak_violence_prob": round(self.peak_violence_prob, 4),
            "fps": round(self.fps, 2),
            "execution_time_sec": round(self.execution_time_sec, 3),
            "window_timeline": self.window_timeline,
        }

    def save_json(self, output_path: str) -> None:
        """Export analysis report to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


class SafeTVPipeline:
    """
    Main processing engine orchestrating video windowing, model forward passes,
    temporal prediction filtering, and incident alerting.
    """

    def __init__(self, config: Optional[SafeTVConfig] = None):
        self.config = config or SafeTVConfig()
        self.extractor = VideoWindowExtractor(self.config.extraction)
        self.model = VideoActionClassifier(self.config.model)
        self.smoother = TemporalSmoother(self.config.smoothing)
        self.incident_classifier = IncidentClassifier(self.config.incident)

    def reset(self) -> None:
        """Reset temporal state and incident trackers."""
        self.smoother.reset()
        self.incident_classifier.reset()

    def process_frames(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0,
        source_name: str = "frame_sequence"
    ) -> PipelineReport:
        """
        Process an in-memory sequence of RGB uint8 video frames.
        """
        self.reset()
        start_exec = time.time()
        timeline: List[WindowResult] = []
        raw_probs: List[float] = []
        smoothed_probs: List[float] = []

        window_idx = 0
        last_end_time = 0.0

        for window in self.extractor.extract_from_frames(frames, fps=fps):
            # 1. Model inference
            preds = self.model.predict_window(window.tensor)
            raw_prob = float(preds["violence_prob"])

            # 2. Temporal smoothing
            smoothed_prob = self.smoother.update(raw_prob)

            # 3. Incident classification
            self.incident_classifier.process_window(
                smoothed_prob=smoothed_prob,
                timestamp_sec=window.center_time_sec,
                frame_idx=window.end_frame
            )

            is_active = self.incident_classifier.is_incident_active
            label = "VIOLENCE" if smoothed_prob >= self.config.incident.onset_threshold else "NORMAL"

            res = WindowResult(
                window_idx=window_idx,
                start_time_sec=window.start_time_sec,
                end_time_sec=window.end_time_sec,
                center_time_sec=window.center_time_sec,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
                raw_violence_prob=raw_prob,
                smoothed_violence_prob=smoothed_prob,
                predicted_label=label,
                incident_active=is_active,
            )
            timeline.append(res)
            raw_probs.append(raw_prob)
            smoothed_probs.append(smoothed_prob)

            last_end_time = max(last_end_time, window.end_time_sec)
            window_idx += 1

        # Finalize open incidents
        final_incidents = self.incident_classifier.finalize()
        exec_duration = time.time() - start_exec

        peak_val = max(smoothed_probs) if smoothed_probs else 0.0
        mean_val = float(np.mean(smoothed_probs)) if smoothed_probs else 0.0

        return PipelineReport(
            video_source=source_name,
            total_duration_sec=last_end_time,
            total_windows_processed=len(timeline),
            incidents_detected=len(final_incidents),
            incidents=[inc.to_dict() for inc in final_incidents],
            mean_violence_prob=mean_val,
            peak_violence_prob=peak_val,
            fps=fps,
            execution_time_sec=exec_duration,
            window_timeline=[w.to_dict() for w in timeline],
        )

    def process_file(
        self,
        video_path: str,
        max_duration_sec: Optional[float] = None
    ) -> PipelineReport:
        """
        Process a video file on disk.
        """
        self.reset()
        start_exec = time.time()
        timeline: List[WindowResult] = []
        raw_probs: List[float] = []
        smoothed_probs: List[float] = []

        window_idx = 0
        last_end_time = 0.0
        fps_detected = 30.0

        for window in self.extractor.extract_from_file(video_path, max_duration_sec=max_duration_sec):
            fps_detected = window.fps
            preds = self.model.predict_window(window.tensor)
            raw_prob = float(preds["violence_prob"])
            smoothed_prob = self.smoother.update(raw_prob)

            self.incident_classifier.process_window(
                smoothed_prob=smoothed_prob,
                timestamp_sec=window.center_time_sec,
                frame_idx=window.end_frame
            )

            is_active = self.incident_classifier.is_incident_active
            label = "VIOLENCE" if smoothed_prob >= self.config.incident.onset_threshold else "NORMAL"

            res = WindowResult(
                window_idx=window_idx,
                start_time_sec=window.start_time_sec,
                end_time_sec=window.end_time_sec,
                center_time_sec=window.center_time_sec,
                start_frame=window.start_frame,
                end_frame=window.end_frame,
                raw_violence_prob=raw_prob,
                smoothed_violence_prob=smoothed_prob,
                predicted_label=label,
                incident_active=is_active,
            )
            timeline.append(res)
            raw_probs.append(raw_prob)
            smoothed_probs.append(smoothed_prob)

            last_end_time = max(last_end_time, window.end_time_sec)
            window_idx += 1

        final_incidents = self.incident_classifier.finalize()
        exec_duration = time.time() - start_exec

        peak_val = max(smoothed_probs) if smoothed_probs else 0.0
        mean_val = float(np.mean(smoothed_probs)) if smoothed_probs else 0.0

        return PipelineReport(
            video_source=video_path,
            total_duration_sec=last_end_time,
            total_windows_processed=len(timeline),
            incidents_detected=len(final_incidents),
            incidents=[inc.to_dict() for inc in final_incidents],
            mean_violence_prob=mean_val,
            peak_violence_prob=peak_val,
            fps=fps_detected,
            execution_time_sec=exec_duration,
            window_timeline=[w.to_dict() for w in timeline],
        )
