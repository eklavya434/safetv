"""
Video frame extraction and temporal window slicing module.
Produces normalized 4D spatiotemporal tensors (C, T, H, W) for action recognition models.
"""

import os
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from .config import VideoExtractionConfig
from .preprocessing import standardize_frame_sequence


@dataclass
class VideoWindow:
    """Represents an extracted spatiotemporal window with temporal metadata."""
    tensor: np.ndarray             # Float32 array shape (C, T, H, W) normalized
    raw_frames: List[np.ndarray]   # Original unnormalized uint8 RGB frames for visualization
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    center_time_sec: float
    fps: float


class VideoWindowExtractor:
    """
    Extracts sliding spatiotemporal windows from video files or frame sequences.
    Applies temporal striding, spatial resizing, and RGB channel normalization.
    """

    def __init__(self, config: Optional[VideoExtractionConfig] = None):
        self.config = config or VideoExtractionConfig()

    def _resize_and_normalize(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Resize RGB frames to (target_height, target_width) and normalize channels to (C, T, H, W)
        using the unified canonical preprocessor.
        """
        return standardize_frame_sequence(
            frames_rgb=frames,
            target_hw=(self.config.target_height, self.config.target_width),
            mean=self.config.normalize_mean,
            std=self.config.normalize_std,
            crop_mode="resize",
        )

    def extract_from_frames(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0
    ) -> Generator[VideoWindow, None, None]:
        """
        Extract sliding windows from a list of RGB uint8 frames.
        Window size represents effective temporal span = config.window_size * config.temporal_stride
        """
        total_frames = len(frames)
        required_span = (self.config.window_size - 1) * self.config.temporal_stride + 1

        if total_frames < required_span:
            # Video shorter than minimum required span; pad or repeat last frame
            if total_frames == 0:
                return
            padding = [frames[-1]] * (required_span - total_frames)
            frames = list(frames) + padding
            total_frames = len(frames)

        step = max(1, self.config.window_step)
        
        for start_idx in range(0, total_frames - required_span + 1, step):
            sampled_indices = [
                start_idx + i * self.config.temporal_stride
                for i in range(self.config.window_size)
            ]
            
            window_raw_frames = [frames[i] for i in sampled_indices]
            window_tensor = self._resize_and_normalize(window_raw_frames)
            
            end_idx = sampled_indices[-1]
            start_time = start_idx / fps
            end_time = end_idx / fps
            center_time = (start_time + end_time) / 2.0

            yield VideoWindow(
                tensor=window_tensor,
                raw_frames=window_raw_frames,
                start_frame=start_idx,
                end_frame=end_idx,
                start_time_sec=start_time,
                end_time_sec=end_time,
                center_time_sec=center_time,
                fps=fps,
            )

    def extract_from_file(
        self,
        video_path: str,
        max_duration_sec: Optional[float] = None
    ) -> Generator[VideoWindow, None, None]:
        """
        Streamingly extracts sliding windows directly from a video file with O(1) memory.
        Maintains a rolling temporal window buffer instead of loading the entire video into RAM.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Failed to open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0

            required_span = (self.config.window_size - 1) * self.config.temporal_stride + 1
            step = max(1, self.config.window_step)
            frame_limit = int(max_duration_sec * fps) if max_duration_sec else None

            # Streaming rolling window frame buffer (max length bounded to required_span + step)
            buffer: List[np.ndarray] = []
            frame_indices: List[int] = []
            count = 0

            while True:
                ret, bgr_frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                # Resize once upon decode for minimal memory footprint (112x112 is ~37KB vs 6.2MB)
                resized_frame = cv2.resize(
                    rgb_frame,
                    (self.config.target_width, self.config.target_height),
                    interpolation=cv2.INTER_LINEAR
                )
                buffer.append(resized_frame)
                frame_indices.append(count)
                count += 1

                if len(buffer) >= required_span:
                    sampled = [buffer[i * self.config.temporal_stride] for i in range(self.config.window_size)]
                    window_tensor = self._resize_and_normalize(sampled)

                    start_f = frame_indices[0]
                    end_f = frame_indices[required_span - 1]
                    start_t = start_f / fps
                    end_t = end_f / fps
                    center_t = (start_t + end_t) / 2.0

                    yield VideoWindow(
                        tensor=window_tensor,
                        raw_frames=[],
                        start_frame=start_f,
                        end_frame=end_f,
                        start_time_sec=start_t,
                        end_time_sec=end_t,
                        center_time_sec=center_t,
                        fps=fps,
                    )

                    # Advance rolling buffer by step
                    buffer = buffer[step:]
                    frame_indices = frame_indices[step:]

                if frame_limit and count >= frame_limit:
                    break

            cap.release()

        except ImportError:
            raise RuntimeError(
                "OpenCV (cv2) is required to read video files directly. "
                "Install via 'pip install opencv-python-headless' or pass frame lists."
            )
