"""
Video integrity inspection and duplicate detection module for SafeTV.
Validates decodability, extracts temporal and spatial metadata, and flags corrupt or unreadable files.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class VideoMetadata:
    """Metadata representing the physical and temporal properties of a video file."""
    file_path: str
    file_name: str
    file_size_bytes: int
    duration_sec: float
    total_frames: int
    fps: float
    width: int
    height: int
    is_valid: bool
    error_message: Optional[str] = None
    fingerprint_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size_bytes": self.file_size_bytes,
            "duration_sec": round(self.duration_sec, 2),
            "total_frames": self.total_frames,
            "fps": round(self.fps, 2),
            "width": self.width,
            "height": self.height,
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "fingerprint_hash": self.fingerprint_hash,
        }


class VideoInspector:
    """
    Inspects video files for decodability, duration, frame count, corruption, and near-duplicate hashing.
    """

    def __init__(self, supported_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")):
        self.supported_extensions = supported_extensions

    def inspect_file(self, video_path: str, check_frames_sample: int = 10) -> VideoMetadata:
        """
        Inspects a video file, reads header metadata, and decodes sample frames to detect corruption.
        """
        file_name = os.path.basename(video_path)

        if not os.path.exists(video_path):
            return VideoMetadata(
                file_path=video_path,
                file_name=file_name,
                file_size_bytes=0,
                duration_sec=0.0,
                total_frames=0,
                fps=0.0,
                width=0,
                height=0,
                is_valid=False,
                error_message=f"File not found: {video_path}",
            )

        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return VideoMetadata(
                file_path=video_path,
                file_name=file_name,
                file_size_bytes=0,
                duration_sec=0.0,
                total_frames=0,
                fps=0.0,
                width=0,
                height=0,
                is_valid=False,
                error_message="File is empty (0 bytes)",
            )

        ext = os.path.splitext(video_path)[1].lower()
        if ext not in self.supported_extensions:
            return VideoMetadata(
                file_path=video_path,
                file_name=file_name,
                file_size_bytes=file_size,
                duration_sec=0.0,
                total_frames=0,
                fps=0.0,
                width=0,
                height=0,
                is_valid=False,
                error_message=f"Unsupported extension: {ext}",
            )

        if cv2 is None:
            return VideoMetadata(
                file_path=video_path,
                file_name=file_name,
                file_size_bytes=file_size,
                duration_sec=0.0,
                total_frames=0,
                fps=0.0,
                width=0,
                height=0,
                is_valid=False,
                error_message="OpenCV (cv2) is not installed",
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VideoMetadata(
                file_path=video_path,
                file_name=file_name,
                file_size_bytes=file_size,
                duration_sec=0.0,
                total_frames=0,
                fps=0.0,
                width=0,
                height=0,
                is_valid=False,
                error_message="Failed to open video stream via OpenCV",
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        if total_frames <= 0 or width <= 0 or height <= 0:
            cap.release()
            return VideoMetadata(
                file_path=video_path,
                file_name=file_name,
                file_size_bytes=file_size,
                duration_sec=0.0,
                total_frames=max(0, total_frames),
                fps=fps,
                width=max(0, width),
                height=max(0, height),
                is_valid=False,
                error_message="Invalid video dimensions or zero frame count",
            )

        duration_sec = total_frames / fps

        # Sample frame decoding check to catch truncated or corrupted bitstreams
        stride = max(1, total_frames // max(1, check_frames_sample))
        read_success_count = 0
        sample_thumbs = []

        for f_idx in range(0, min(total_frames, check_frames_sample * stride), stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                read_success_count += 1
                # Save tiny 16x16 grayscale thumbnail for duplicate fingerprinting
                gray_tiny = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (16, 16))
                sample_thumbs.append(gray_tiny)
            else:
                break

        cap.release()

        if read_success_count == 0:
            return VideoMetadata(
                file_path=video_path,
                file_name=file_name,
                file_size_bytes=file_size,
                duration_sec=duration_sec,
                total_frames=total_frames,
                fps=fps,
                width=width,
                height=height,
                is_valid=False,
                error_message="Failed to decode sample frames (corrupt stream)",
            )

        # Compute spatiotemporal fingerprint hash from thumbnails
        fingerprint = None
        if sample_thumbs:
            stacked = np.stack(sample_thumbs, axis=0) # (K, 16, 16)
            # Binary threshold hash relative to mean intensity
            bits = (stacked > np.mean(stacked)).astype(np.uint8).flatten()
            fingerprint = f"{np.sum(bits):04d}_{hash(bits.tobytes()) & 0xFFFFFFFF:08x}"

        return VideoMetadata(
            file_path=video_path,
            file_name=file_name,
            file_size_bytes=file_size,
            duration_sec=duration_sec,
            total_frames=total_frames,
            fps=fps,
            width=width,
            height=height,
            is_valid=True,
            error_message=None,
            fingerprint_hash=fingerprint,
        )

    def scan_directory(self, root_dir: str) -> Tuple[List[VideoMetadata], List[VideoMetadata]]:
        """
        Recursively scans directory for videos, returning (valid_videos, invalid_videos).
        """
        valid_list = []
        invalid_list = []

        if not os.path.exists(root_dir):
            return valid_list, invalid_list

        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.supported_extensions:
                    full_p = os.path.join(dirpath, fname)
                    meta = self.inspect_file(full_p)
                    if meta.is_valid:
                        valid_list.append(meta)
                    else:
                        invalid_list.append(meta)

        return valid_list, invalid_list
