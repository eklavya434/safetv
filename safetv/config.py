"""
Configuration dataclasses for SafeTV Violence Detection Engine.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class VideoExtractionConfig:
    """Configuration for sliding spatiotemporal video window extraction."""
    window_size: int = 16          # Number of frames per temporal window (T)
    temporal_stride: int = 2       # Sample every n-th frame (downsampling in time)
    window_step: int = 8           # Stride between consecutive sliding windows (overlap = window_size - window_step)
    target_height: int = 112       # Standard action recognition spatial height
    target_width: int = 112        # Standard action recognition spatial width
    normalize_mean: Tuple[float, float, float] = (0.43216, 0.394666, 0.37645)  # Kinetics-400 RGB mean
    normalize_std: Tuple[float, float, float] = (0.22803, 0.22145, 0.216989)   # Kinetics-400 RGB std


@dataclass
class ModelConfig:
    """Configuration for video action recognition backbone."""
    architecture: str = "r2plus1d_18"  # Supported: "r2plus1d_18", "mc3_18"
    num_classes: int = 2               # 0: Normal Activity, 1: Potential Violence
    pretrained: bool = True
    checkpoint_path: Optional[str] = "models/violence_detector/best_model.pth"
    device: str = "cpu"                # Target device: "cpu", "cuda", or "directml"
    num_threads: int = 6               # CPU intra-op parallelism for Intel i5-12450H


@dataclass
class EvidenceConfig:
    """Configuration for incident evidence clip slicing and archival."""
    evidence_dir: str = "evidence"         # Output directory for video evidence clips
    pre_buffer_sec: float = 2.0            # Pre-incident buffer duration in seconds
    post_buffer_sec: float = 2.0           # Post-incident buffer duration in seconds
    save_evidence_clips: bool = True       # Whether to automatically slice and save MP4 clips
    save_incident_metadata: bool = True    # Whether to write JSON metadata per incident
    video_codec: str = "mp4v"              # Codec for OpenCV VideoWriter ('mp4v' or 'avc1')


@dataclass
class TemporalSmoothingConfig:
    """Configuration for temporal probability smoothing to eliminate single-frame jitter."""
    filter_type: str = "ema"          # "ema" (Exponential Moving Average) or "sliding_window"
    ema_alpha: float = 0.35           # Smoothing factor (0 < alpha <= 1; lower = smoother, higher = more responsive)
    window_length: int = 5            # Number of historical window predictions to average
    median_kernel_size: int = 3       # Optional median filter kernel to filter impulse anomalies


@dataclass
class IncidentClassifierConfig:
    """Configuration for dual-threshold hysteresis incident classification."""
    onset_threshold: float = 0.60     # Threshold to trigger an incident (violence alert)
    offset_threshold: float = 0.40    # Threshold to clear an active incident
    min_sustained_windows: int = 2    # Minimum consecutive windows above onset before firing an alert
    cool_off_windows: int = 2         # Minimum consecutive windows below offset before concluding incident
    severity_thresholds: dict = field(
        default_factory=lambda: {
            "low": 0.50,
            "medium": 0.70,
            "high": 0.85,
        }
    )


@dataclass
class SafeTVConfig:
    """Master configuration uniting video, model, smoothing, incident classification, and evidence."""
    extraction: VideoExtractionConfig = field(default_factory=VideoExtractionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    smoothing: TemporalSmoothingConfig = field(default_factory=TemporalSmoothingConfig)
    incident: IncidentClassifierConfig = field(default_factory=IncidentClassifierConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
