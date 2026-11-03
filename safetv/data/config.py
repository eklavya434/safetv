"""
Dataset and Preprocessing Configuration for SafeTV.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DatasetConfig:
    """Configuration for SafeTV video dataset creation, preprocessing, and splitting."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    annotations_dir: str = "data/annotations"

    supported_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    # Spatiotemporal sampling
    num_frames: int = 16            # Temporal sequence length (T)
    temporal_stride: int = 2        # Sample every n-th frame
    target_height: int = 112        # Spatial height (H)
    target_width: int = 112         # Spatial width (W)
    fps_target: float = 30.0        # Target frame rate for standardization

    # Normalization (Kinetics-400 RGB mean & standard deviation)
    normalize_mean: Tuple[float, float, float] = (0.43216, 0.394666, 0.37645)
    normalize_std: Tuple[float, float, float] = (0.22803, 0.22145, 0.216989)

    # Splitting parameters (strictly by original video to avoid frame leakage)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # Binary label mapping
    class_to_idx: Dict[str, int] = field(
        default_factory=lambda: {
            "non_violence": 0,
            "violence": 1,
        }
    )
    idx_to_class: Dict[int, str] = field(
        default_factory=lambda: {
            0: "non_violence",
            1: "violence",
        }
    )

    # Granular categories for audit and metadata tracking
    violence_subcategories: List[str] = field(
        default_factory=lambda: [
            "punching",
            "kicking",
            "physical_fights",
            "pushing_contact",
            "group_altercation",
            "assault",
        ]
    )
    non_violence_subcategories: List[str] = field(
        default_factory=lambda: [
            "normal_walking",
            "running",
            "talking",
            "standing",
            "playing",
            "exercising",
            "dancing",
            "hugging",
            "celebration",
            "sports",
            "helping",
        ]
    )
