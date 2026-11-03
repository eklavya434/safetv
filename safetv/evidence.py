"""
SafeTV Evidence Archival and Forensic Clip Extraction Engine.
Extracts, buffers, cuts, and exports MP4 video evidence clips alongside
structured JSON forensic audit trails with model checkpoint SHA-256 verification.
"""

import datetime
import json
import os
import re
import uuid
from typing import List, Optional, Tuple, Union

import numpy as np

from safetv.config import EvidenceConfig
from safetv.incident_classifier import Incident


class EvidenceExtractor:
    """
    Slices and archives spatiotemporal evidence video clips with pre- and post-incident
    temporal buffering for human security operator review and chain-of-custody logging.
    """

    def __init__(self, config: Optional[EvidenceConfig] = None):
        self.config = config or EvidenceConfig()
        os.makedirs(self.config.evidence_dir, exist_ok=True)

    def _slugify(self, text: str) -> str:
        """Sanitize string for filesystem paths."""
        slug = re.sub(r"[^\w\-_.]", "_", text)
        return slug[:32]

    def _get_target_paths(self, incident: Incident, source_name: str) -> Tuple[str, str]:
        """Generate formatted unique destination paths for MP4 evidence clip and JSON metadata."""
        timestamp_slug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        source_slug = self._slugify(os.path.splitext(os.path.basename(source_name))[0])
        unique_suffix = uuid.uuid4().hex[:6]
        filename_base = f"INC-{incident.incident_id:04d}_{source_slug}_{timestamp_slug}_{unique_suffix}"

        clip_path = os.path.join(self.config.evidence_dir, f"{filename_base}.mp4")
        meta_path = os.path.join(self.config.evidence_dir, f"{filename_base}_metadata.json")
        return clip_path, meta_path

    def _write_metadata(
        self,
        meta_path: str,
        incident: Incident,
        source_name: str,
        clip_path: str,
        clip_start_sec: float,
        clip_end_sec: float,
        fps: float,
        frame_count: int,
        resolution: Tuple[int, int] = (0, 0),
        checkpoint_hash: Optional[str] = None,
    ) -> None:
        """Export standardized incident forensic metadata record."""
        record = {
            "incident_id": incident.incident_id,
            "video_source": source_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "incident_temporal_window": {
                "start_time_sec": round(incident.start_time_sec, 2),
                "end_time_sec": round(incident.end_time_sec, 2),
                "duration_sec": round(incident.duration_sec, 2),
                "start_frame": incident.start_frame,
                "end_frame": incident.end_frame,
            },
            "evidence_clip": {
                "clip_path": os.path.abspath(clip_path),
                "clip_filename": os.path.basename(clip_path),
                "clip_start_sec": round(clip_start_sec, 2),
                "clip_end_sec": round(clip_end_sec, 2),
                "clip_duration_sec": round(max(0.0, clip_end_sec - clip_start_sec), 2),
                "pre_incident_buffer_sec": self.config.pre_buffer_sec,
                "post_incident_buffer_sec": self.config.post_buffer_sec,
                "fps": round(fps, 2),
                "resolution": f"{resolution[0]}x{resolution[1]}",
                "total_clip_frames": frame_count,
            },
            "detection_metrics": {
                "peak_probability": round(incident.peak_probability, 4),
                "mean_probability": round(incident.mean_probability, 4),
                "severity": incident.severity,
                "model_checkpoint_hash": checkpoint_hash or "unknown",
            },
            "status": "PENDING_OPERATOR_REVIEW",
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

    def extract_clip_from_frames(
        self,
        frames: List[np.ndarray],
        fps: float,
        incident: Incident,
        source_name: str = "stream",
        checkpoint_hash: Optional[str] = None,
    ) -> Optional[str]:
        """
        Extract buffered incident clip from in-memory RGB frame sequence.
        """
        if not self.config.save_evidence_clips or not frames:
            return None

        total_frames = len(frames)
        total_duration = total_frames / fps if fps > 0 else 0.0

        # Calculate buffered time range
        clip_start_sec = max(0.0, incident.start_time_sec - self.config.pre_buffer_sec)
        clip_end_sec = min(total_duration, incident.end_time_sec + self.config.post_buffer_sec)
        if clip_end_sec <= clip_start_sec:
            clip_end_sec = total_duration

        start_frame_idx = max(0, int(clip_start_sec * fps))
        end_frame_idx = min(total_frames, int(clip_end_sec * fps) + 1)
        subframes = frames[start_frame_idx:end_frame_idx]

        if not subframes:
            # Fallback to all available frames if slice would be empty
            subframes = frames

        clip_path, meta_path = self._get_target_paths(incident, source_name)
        h, w = subframes[0].shape[:2]

        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
            writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))

            for frame in subframes:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 and frame.shape[2] == 3 else frame
                writer.write(bgr)

            writer.release()

            if self.config.save_incident_metadata:
                self._write_metadata(
                    meta_path=meta_path,
                    incident=incident,
                    source_name=source_name,
                    clip_path=clip_path,
                    clip_start_sec=clip_start_sec,
                    clip_end_sec=clip_end_sec,
                    fps=fps,
                    frame_count=len(subframes),
                    resolution=(w, h),
                    checkpoint_hash=checkpoint_hash,
                )
                incident.metadata_path = meta_path

            incident.evidence_path = clip_path
            return clip_path

        except Exception as e:
            print(f"[!] Warning: Evidence clip writing failed ({e}). Saving metadata only.")
            if self.config.save_incident_metadata:
                self._write_metadata(
                    meta_path=meta_path,
                    incident=incident,
                    source_name=source_name,
                    clip_path=clip_path,
                    clip_start_sec=clip_start_sec,
                    clip_end_sec=clip_end_sec,
                    fps=fps,
                    frame_count=len(subframes),
                    resolution=(w, h),
                    checkpoint_hash=checkpoint_hash,
                )
                incident.metadata_path = meta_path
            return None

    def extract_clip_from_file(
        self,
        video_path: str,
        incident: Incident,
        checkpoint_hash: Optional[str] = None,
    ) -> Optional[str]:
        """
        Extract buffered incident clip directly from an on-disk video file.
        """
        if not self.config.save_evidence_clips or not os.path.exists(video_path):
            return None

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = total_frames / fps if fps > 0 else 0.0

            clip_start_sec = max(0.0, incident.start_time_sec - self.config.pre_buffer_sec)
            clip_end_sec = min(total_duration, incident.end_time_sec + self.config.post_buffer_sec)
            if clip_end_sec <= clip_start_sec:
                clip_end_sec = total_duration

            start_frame_idx = max(0, int(clip_start_sec * fps))
            end_frame_idx = min(total_frames, int(clip_end_sec * fps) + 1) if total_frames > 0 else int(clip_end_sec * fps) + 1

            clip_path, meta_path = self._get_target_paths(incident, video_path)

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
            writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            current_frame = start_frame_idx
            written_count = 0

            while current_frame <= end_frame_idx:
                ret, bgr_frame = cap.read()
                if not ret:
                    break
                writer.write(bgr_frame)
                written_count += 1
                current_frame += 1

            cap.release()
            writer.release()

            if written_count > 0:
                if self.config.save_incident_metadata:
                    self._write_metadata(
                        meta_path=meta_path,
                        incident=incident,
                        source_name=video_path,
                        clip_path=clip_path,
                        clip_start_sec=clip_start_sec,
                        clip_end_sec=clip_end_sec,
                        fps=fps,
                        frame_count=written_count,
                        resolution=(w, h),
                        checkpoint_hash=checkpoint_hash,
                    )
                    incident.metadata_path = meta_path

                incident.evidence_path = clip_path
                return clip_path

            return None

        except Exception as e:
            print(f"[!] Warning: Evidence extraction from file failed ({e}).")
            return None
