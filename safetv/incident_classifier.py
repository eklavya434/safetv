"""
Incident classification state machine using dual-threshold hysteresis.
Distinguishes genuine sustained violence incidents from transient high-energy motions.
Enforces precise onset window timestamping and excludes cooling-off periods from duration.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .config import IncidentClassifierConfig


@dataclass
class Incident:
    """Represents a detected violent incident with spatiotemporal timestamps and severity."""
    incident_id: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    start_frame: int
    end_frame: int
    peak_probability: float
    mean_probability: float
    severity: str
    sample_timestamps: List[float] = field(default_factory=list)
    sample_probabilities: List[float] = field(default_factory=list)
    evidence_path: Optional[str] = None
    metadata_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "start_time_sec": round(self.start_time_sec, 2),
            "end_time_sec": round(self.end_time_sec, 2),
            "duration_sec": round(self.duration_sec, 2),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "peak_probability": round(self.peak_probability, 4),
            "mean_probability": round(self.mean_probability, 4),
            "severity": self.severity,
            "evidence_path": self.evidence_path,
            "metadata_path": self.metadata_path,
        }


class IncidentClassifier:
    """
    Stateful incident detector implementing hysteresis state transitions:
    IDLE -> POTENTIAL_INCIDENT -> ACTIVE_INCIDENT -> COOLING_OFF -> IDLE.
    Accurately attributes incident onset to the first qualifying window.
    """

    def __init__(self, config: Optional[IncidentClassifierConfig] = None):
        self.config = config or IncidentClassifierConfig()
        self.reset()

    def reset(self) -> None:
        """Reset state machine."""
        self._is_active: bool = False
        self._consecutive_onset: int = 0
        self._consecutive_offset: int = 0
        self._current_incident_id: int = 0

        # Pending onset tracking (buffers the onset windows before formal confirmation)
        self._pending_onset_timestamps: List[float] = []
        self._pending_onset_frames: List[int] = []
        self._pending_onset_probs: List[float] = []

        # Active incident tracking
        self._active_window_probs: List[float] = []
        self._active_timestamps: List[float] = []
        self._active_start_time: float = 0.0
        self._active_start_frame: int = 0
        self._active_end_time: float = 0.0
        self._active_end_frame: int = 0
        self._completed_incidents: List[Incident] = []

    def _determine_severity(self, peak_prob: float) -> str:
        """Categorize severity based on peak smoothed probability."""
        if peak_prob >= self.config.severity_thresholds.get("high", 0.85):
            return "CRITICAL"
        elif peak_prob >= self.config.severity_thresholds.get("medium", 0.70):
            return "HIGH"
        elif peak_prob >= self.config.severity_thresholds.get("low", 0.50):
            return "MEDIUM"
        return "LOW"

    def process_window(
        self,
        smoothed_prob: float,
        timestamp_sec: float,
        frame_idx: int
    ) -> Optional[Incident]:
        """
        Process a single smoothed window prediction.
        Returns a completed Incident when an incident transitions from ACTIVE to RESOLVED.
        """
        completed_incident = None

        if not self._is_active:
            # Check for onset condition
            if smoothed_prob >= self.config.onset_threshold:
                self._pending_onset_timestamps.append(timestamp_sec)
                self._pending_onset_frames.append(frame_idx)
                self._pending_onset_probs.append(smoothed_prob)
                self._consecutive_onset += 1

                if self._consecutive_onset >= self.config.min_sustained_windows:
                    # Formally transition into ACTIVE_INCIDENT
                    # Key fix: Incident start corresponds to the FIRST qualifying onset window
                    self._is_active = True
                    self._consecutive_offset = 0
                    self._current_incident_id += 1

                    self._active_start_time = self._pending_onset_timestamps[0]
                    self._active_start_frame = self._pending_onset_frames[0]
                    self._active_end_time = timestamp_sec
                    self._active_end_frame = frame_idx
                    self._active_window_probs = list(self._pending_onset_probs)
                    self._active_timestamps = list(self._pending_onset_timestamps)

                    self._pending_onset_timestamps = []
                    self._pending_onset_frames = []
                    self._pending_onset_probs = []
            else:
                # Reset pending onset if threshold is not sustained
                self._consecutive_onset = 0
                self._pending_onset_timestamps = []
                self._pending_onset_frames = []
                self._pending_onset_probs = []
        else:
            # Active incident tracking
            self._active_window_probs.append(smoothed_prob)
            self._active_timestamps.append(timestamp_sec)

            if smoothed_prob > self.config.offset_threshold:
                # Still sustained violence: update active end timestamp to this qualifying active window
                self._active_end_time = timestamp_sec
                self._active_end_frame = frame_idx
                self._consecutive_offset = 0
            else:
                # Cooling off: probability has dropped below threshold
                self._consecutive_offset += 1
                if self._consecutive_offset >= self.config.cool_off_windows:
                    # Transition back to IDLE; conclude incident
                    self._is_active = False
                    self._consecutive_onset = 0
                    self._consecutive_offset = 0

                    peak = max(self._active_window_probs) if self._active_window_probs else smoothed_prob
                    mean_val = float(np.mean(self._active_window_probs)) if self._active_window_probs else smoothed_prob
                    # Duration is strictly between start and last active window (excluding cooling-off windows)
                    duration = max(0.0, self._active_end_time - self._active_start_time)
                    severity = self._determine_severity(peak)

                    incident = Incident(
                        incident_id=self._current_incident_id,
                        start_time_sec=self._active_start_time,
                        end_time_sec=self._active_end_time,
                        duration_sec=duration,
                        start_frame=self._active_start_frame,
                        end_frame=self._active_end_frame,
                        peak_probability=peak,
                        mean_probability=mean_val,
                        severity=severity,
                        sample_timestamps=list(self._active_timestamps),
                        sample_probabilities=list(self._active_window_probs),
                    )
                    self._completed_incidents.append(incident)
                    completed_incident = incident

        return completed_incident

    def finalize(self) -> List[Incident]:
        """
        Close any remaining open incident at the end of video processing.
        Returns the full list of detected incidents.
        """
        if self._is_active and self._active_window_probs:
            peak = max(self._active_window_probs)
            mean_val = float(np.mean(self._active_window_probs))
            duration = max(0.0, self._active_end_time - self._active_start_time)
            severity = self._determine_severity(peak)

            incident = Incident(
                incident_id=self._current_incident_id,
                start_time_sec=self._active_start_time,
                end_time_sec=self._active_end_time,
                duration_sec=duration,
                start_frame=self._active_start_frame,
                end_frame=self._active_end_frame,
                peak_probability=peak,
                mean_probability=mean_val,
                severity=severity,
                sample_timestamps=list(self._active_timestamps),
                sample_probabilities=list(self._active_window_probs),
            )
            self._completed_incidents.append(incident)
            self._is_active = False

        return list(self._completed_incidents)

    @property
    def is_incident_active(self) -> bool:
        """Returns True if the system is currently inside an active incident state."""
        return self._is_active

    @property
    def completed_incidents(self) -> List[Incident]:
        """List of all finalized incidents."""
        return self._completed_incidents
