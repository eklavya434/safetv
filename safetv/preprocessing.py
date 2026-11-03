"""
Unified spatiotemporal preprocessing module for SafeTV.
Provides a single canonical preprocessing implementation shared by training and inference.
Guarantees mathematical parity in color space, pixel scaling, normalization, and tensor shapes.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

# Kinetics-400 video normalization constants
KINETICS_MEAN: Tuple[float, float, float] = (0.43216, 0.394666, 0.37645)
KINETICS_STD: Tuple[float, float, float] = (0.22803, 0.22145, 0.216989)


def standardize_frame_sequence(
    frames_rgb: List[np.ndarray],
    target_hw: Tuple[int, int] = (112, 112),
    mean: Tuple[float, float, float] = KINETICS_MEAN,
    std: Tuple[float, float, float] = KINETICS_STD,
    crop_mode: str = "resize",
) -> np.ndarray:
    """
    Unified spatiotemporal preprocessing function for action recognition.
    
    Args:
        frames_rgb: Sequence of T RGB uint8 images (H, W, 3).
        target_hw: Output resolution (target_height, target_width).
        mean: Channel means for normalization.
        std: Channel standard deviations for normalization.
        crop_mode: 'resize' for bilinear stretch or 'center_crop' for aspect-preserving crop.

    Returns:
        Normalized float32 array of shape (C, T, H, W) where C=3.
    """
    if not frames_rgb:
        raise ValueError("Cannot preprocess an empty sequence of frames.")

    target_h, target_w = target_hw
    mean_arr = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    processed_tensors = []
    for frame in frames_rgb:
        if frame is None or frame.size == 0:
            raise ValueError("Encountered empty or None frame during preprocessing.")

        h, w = frame.shape[:2]

        if crop_mode == "center_crop":
            # Aspect-preserving resize shorter dimension to 128 then center crop to target
            scale = 128.0 / min(h, w)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            if cv2 is not None:
                scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled = np.array(Image.fromarray(frame).resize((new_w, new_h), resample=Image.Resampling.BILINEAR))
            y_start = max(0, (new_h - target_h) // 2)
            x_start = max(0, (new_w - target_w) // 2)
            resized = scaled[y_start : y_start + target_h, x_start : x_start + target_w]
        else:
            if (w, h) != (target_w, target_h):
                if cv2 is not None:
                    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                else:
                    pil_img = Image.fromarray(frame)
                    resized = np.array(pil_img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR))
            else:
                resized = frame

        # Float scaling [0.0, 1.0]
        arr = np.array(resized, dtype=np.float32) / 255.0  # (H, W, C)
        # Transpose to (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))
        # Kinetics channel normalization
        arr = (arr - mean_arr) / std_arr
        processed_tensors.append(arr)

    # Stack into (T, C, H, W) -> Transpose to (C, T, H, W)
    t_stack = np.stack(processed_tensors, axis=0)
    t_stack = np.transpose(t_stack, (1, 0, 2, 3))
    return t_stack.astype(np.float32)
