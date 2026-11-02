"""
Temporal prediction smoothing module for SafeTV.
Suppresses high-frequency jitter, transient false alarms, and single-frame flickers.
"""

from typing import List, Optional
import numpy as np

from .config import TemporalSmoothingConfig


class TemporalSmoother:
    """
    Temporal probability smoothing for streaming and offline action recognition.
    Combines Exponential Moving Average (EMA), Rolling Mean, and Median filtering.
    """

    def __init__(self, config: Optional[TemporalSmoothingConfig] = None):
        self.config = config or TemporalSmoothingConfig()
        self.reset()

    def reset(self) -> None:
        """Reset internal history and state buffers."""
        self._history: List[float] = []
        self._ema_val: Optional[float] = None

    def update(self, raw_prob: float) -> float:
        """
        Process a single incoming window probability in real-time streaming mode.
        Returns the smoothed probability score in [0.0, 1.0].
        """
        raw_prob = float(np.clip(raw_prob, 0.0, 1.0))
        self._history.append(raw_prob)

        # Truncate history buffer to save memory
        max_buffer = max(self.config.window_length * 2, 20)
        if len(self._history) > max_buffer:
            self._history.pop(0)

        # 1. Median filtering on recent tail to reject impulse spikes if requested
        k_med = self.config.median_kernel_size
        if k_med > 1 and len(self._history) >= k_med:
            recent_sub = self._history[-k_med:]
            filtered_input = float(np.median(recent_sub))
        else:
            filtered_input = raw_prob

        # 2. Primary smoothing filter
        if self.config.filter_type == "ema":
            if self._ema_val is None:
                self._ema_val = filtered_input
            else:
                alpha = self.config.ema_alpha
                self._ema_val = alpha * filtered_input + (1.0 - alpha) * self._ema_val
            smoothed = self._ema_val

        elif self.config.filter_type == "sliding_window":
            win_len = min(len(self._history), self.config.window_length)
            smoothed = float(np.mean(self._history[-win_len:]))

        else:
            smoothed = filtered_input

        return float(np.clip(smoothed, 0.0, 1.0))

    def smooth_sequence(self, raw_probs: List[float]) -> List[float]:
        """
        Batch smoothing of an entire sequence of probabilities.
        """
        self.reset()
        smoothed_probs = []
        for p in raw_probs:
            smoothed_probs.append(self.update(p))
        return smoothed_probs
