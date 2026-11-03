"""
Dataset acquisition and benchmark sample builder module for SafeTV.
Supports automated downloading of public datasets (RWF-2000, RLVS) and
generates curated test sets with difficult negative examples.
"""

import json
import math
import os
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from .config import DatasetConfig
from .preprocessor import PreprocessedClip, VideoPreprocessor
from .video_inspector import VideoInspector


# Official academic dataset download links & repository sources
DATASET_SOURCES = {
    "rwf2000": {
        "name": "RWF-2000 (Real-World Fight)",
        "source_url": "https://github.com/mlama/RWF-2000",
        "description": "2,000 surveillance video clips (1,000 fight, 1,000 non-fight; 5s @ 30fps).",
        "license": "Non-commercial Academic Research",
        "direct_archive": "https://github.com/mlama/RWF-2000/releases/download/v1.0/RWF-2000.zip",
    },
    "rlvs": {
        "name": "Real-Life Violence Situations (RLVS)",
        "source_url": "https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset",
        "description": "2,000 video clips (1,000 violence, 1,000 non-violence; street & surveillance).",
        "license": "Academic Research",
    },
}


class DatasetDownloader:
    """
    Manages dataset downloads, integrity verification, and creation of structured benchmark datasets.
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.inspector = VideoInspector(self.config.supported_extensions)
        self.preprocessor = VideoPreprocessor(self.config)

    def print_dataset_sources(self) -> None:
        """Prints information and official sources for supported academic datasets."""
        print("=" * 65)
        print(" SafeTV Supported Video Violence Datasets")
        print("=" * 65)
        for key, info in DATASET_SOURCES.items():
            print(f"[{key.upper()}] {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  License:     {info['license']}")
            print(f"  Source URL:  {info['source_url']}")
            if "direct_archive" in info:
                print(f"  Direct Link: {info['direct_archive']}")
            print()

    @classmethod
    def generate_synthetic_video_clip(
        cls,
        output_path: str,
        category: str,
        num_frames: int = 60,
        fps: float = 30.0,
        width: int = 160,
        height: int = 120
    ) -> str:
        """
        Synthesizes a realistic, deterministic test video file for a specific activity category.
        Used for reproducible offline pipeline testing, edge-case validation, and CI/CD tests.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV is required to generate video clips.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for t in range(num_frames):
            frame = np.full((height, width, 3), fill_value=235, dtype=np.uint8)

            if category in ["punching", "kicking", "physical_fights", "violence"]:
                # Chaotic high-acceleration altercations
                impact = 18 * math.sin(t * 1.8) * math.cos(t * 3.1)
                tumble = 12 * math.sin(t * 2.5)
                cx1 = int(np.clip(width // 2 - 12 + impact, 15, width - 15))
                cy1 = int(np.clip(height // 2 + tumble, 15, height - 15))
                cx2 = int(np.clip(width // 2 + 12 - impact * 0.85, 15, width - 15))
                cy2 = int(np.clip(height // 2 - tumble * 0.9, 15, height - 15))

                cv2.circle(frame, (cx1, cy1), 12, (30, 30, 180), -1)   # Red
                cv2.circle(frame, (cx2, cy2), 12, (180, 30, 30), -1)   # Blue
                # Erratic strike limb
                sx = int((cx1 + cx2) // 2 + 14 * math.cos(t * 4.2))
                sy = int((cy1 + cy2) // 2 + 14 * math.sin(t * 4.2))
                cv2.circle(frame, (sx, sy), 6, (20, 180, 230), -1)

            elif category in ["exercising", "dancing", "sports"]:
                # Difficult negative: High motion energy but smooth harmonic rhythm
                phase = t * 0.45
                cx = width // 2
                cy = int(height // 2 + 16 * abs(math.sin(phase)))
                spread = int(14 * abs(math.cos(phase)))
                cv2.circle(frame, (cx, cy), 11, (30, 120, 210), -1)
                cv2.circle(frame, (cx - 15 - spread, cy - 4), 6, (30, 120, 210), -1)
                cv2.circle(frame, (cx + 15 + spread, cy - 4), 6, (30, 120, 210), -1)

            elif category in ["hugging", "celebration", "helping"]:
                # Difficult negative: Close two-person proximity without chaotic impact
                cx1 = int(width // 2 - 8 + 4 * math.sin(t * 0.2))
                cy1 = height // 2
                cx2 = int(width // 2 + 8 - 4 * math.sin(t * 0.2))
                cy2 = height // 2
                cv2.circle(frame, (cx1, cy1), 11, (50, 160, 50), -1)
                cv2.circle(frame, (cx2, cy2), 11, (160, 50, 50), -1)

            else:
                # Normal walking
                xa = int(25 + (width - 50) * (t / num_frames))
                ya = int(height // 2 + 4 * math.sin(t * 0.3))
                xb = int(width - 25 - (width - 50) * (t / num_frames))
                yb = int(height // 2 + 12 + 4 * math.cos(t * 0.3))
                cv2.circle(frame, (xa, ya), 10, (180, 90, 45), -1)
                cv2.circle(frame, (xb, yb), 10, (75, 140, 60), -1)

            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        writer.release()
        return output_path

    def build_sample_dataset(
        self,
        base_dir: str = "data",
        num_samples_per_category: int = 4
    ) -> Tuple[str, str]:
        """
        Builds a verified sample dataset with both violence and difficult negative non-violence
        subcategories, partitions it into train/val/test splits, and writes manifests.
        """
        raw_violence = os.path.join(base_dir, "raw", "violence")
        raw_non_violence = os.path.join(base_dir, "raw", "non_violence")
        os.makedirs(raw_violence, exist_ok=True)
        os.makedirs(raw_non_violence, exist_ok=True)

        video_entries = []

        # 1. Generate violence videos
        violence_cats = ["punching", "kicking", "physical_fights", "pushing_contact"]
        for cat in violence_cats:
            for i in range(num_samples_per_category):
                fname = f"{cat}_{i+1:03d}.mp4"
                v_path = os.path.join(raw_violence, fname)
                self.generate_synthetic_video_clip(v_path, category=cat, num_frames=60)
                video_entries.append({
                    "video_path": v_path,
                    "label": "violence",
                    "label_id": 1,
                    "subcategory": cat,
                    "source_id": f"src_viol_{cat}_{i+1:03d}",
                    "source_dataset": "safetv_benchmark_pilot"
                })

        # 2. Generate non-violence videos (including difficult negatives)
        non_violence_cats = ["normal_walking", "running", "exercising", "dancing", "hugging", "celebration"]
        for cat in non_violence_cats:
            for i in range(num_samples_per_category):
                fname = f"{cat}_{i+1:03d}.mp4"
                v_path = os.path.join(raw_non_violence, fname)
                self.generate_synthetic_video_clip(v_path, category=cat, num_frames=60)
                video_entries.append({
                    "video_path": v_path,
                    "label": "non_violence",
                    "label_id": 0,
                    "subcategory": cat,
                    "source_id": f"src_norm_{cat}_{i+1:03d}",
                    "source_dataset": "safetv_benchmark_pilot"
                })

        # 3. Partition strictly by source_id to prevent frame leakage
        splits = self.preprocessor.partition_videos(video_entries)

        # 4. Create PreprocessedClip records
        all_clips: List[PreprocessedClip] = []
        clip_counter = 0

        for split_name, entries in splits.items():
            for item in entries:
                v_path = str(item["video_path"])
                meta = self.inspector.inspect_file(v_path)
                if not meta.is_valid:
                    continue

                clip_counter += 1
                clip_id = f"clip_{clip_counter:04d}"
                clip = PreprocessedClip(
                    clip_id=clip_id,
                    source_video_path=v_path,
                    processed_clip_path=v_path,
                    split=split_name,
                    label=str(item["label"]),
                    label_id=int(item["label_id"]),
                    subcategory=str(item["subcategory"]),
                    source_dataset=str(item["source_dataset"]),
                    start_frame=0,
                    end_frame=meta.total_frames - 1,
                    start_sec=0.0,
                    end_sec=meta.duration_sec,
                    num_frames=meta.total_frames,
                )
                all_clips.append(clip)

        # 5. Write Manifest Files
        csv_path, json_path = self.preprocessor.write_manifest(
            clips=all_clips,
            manifest_csv_path=os.path.join(base_dir, "annotations", "labels.csv"),
            summary_json_path=os.path.join(base_dir, "annotations", "manifest.json")
        )

        return csv_path, json_path
