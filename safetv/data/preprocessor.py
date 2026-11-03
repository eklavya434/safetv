"""
SafeTV Dataset Preprocessing and Partitioning Module.
Extracts spatiotemporal tensors, enforces video-level split separation (no data leakage),
and writes manifest annotations.
"""

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from .config import DatasetConfig
from .video_inspector import VideoInspector, VideoMetadata
from safetv.preprocessing import standardize_frame_sequence


@dataclass
class PreprocessedClip:
    """Represents a standardized preprocessed video clip ready for training."""
    clip_id: str
    source_video_path: str
    processed_clip_path: Optional[str]
    split: str                          # "train", "val", "test"
    label: str                          # "violence", "non_violence"
    label_id: int                       # 1, 0
    subcategory: str                    # e.g. "punching", "running"
    source_dataset: str                 # e.g. "rwf2000", "safe_tv_pilot"
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    num_frames: int

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "source_video_path": self.source_video_path,
            "processed_clip_path": self.processed_clip_path or "",
            "split": self.split,
            "label": self.label,
            "label_id": self.label_id,
            "subcategory": self.subcategory,
            "source_dataset": self.source_dataset,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "num_frames": self.num_frames,
        }


class VideoPreprocessor:
    """
    Standardizes raw video feeds into uniform spatiotemporal clips,
    partitions datasets strictly by original source video to eliminate frame leakage,
    and exports annotation manifests.
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.inspector = VideoInspector(self.config.supported_extensions)

    def extract_clip_tensor(
        self,
        video_path: str,
        start_frame: int = 0,
        temporal_stride: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Reads a video file and extracts a normalized (C, T, H, W) float32 tensor.
        Uses temporal subsampling and bilinear spatial interpolation.
        """
        stride = temporal_stride or self.config.temporal_stride
        required_span = (self.config.num_frames - 1) * stride + 1

        if cv2 is None or not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        # Determine start index safely
        max_valid_start = max(0, total_frames - required_span)
        start_idx = min(start_frame, max_valid_start)

        frames_rgb = []
        for i in range(self.config.num_frames):
            target_f = min(total_frames - 1, start_idx + i * stride)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None:
                # Do not silently replace corrupted or unreadable frames with zero tensors
                cap.release()
                return None
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)

        cap.release()

        # Normalize and arrange into (C, T, H, W) via unified preprocessor
        return standardize_frame_sequence(
            frames_rgb=frames_rgb,
            target_hw=(self.config.target_height, self.config.target_width),
            mean=tuple(self.config.normalize_mean),
            std=tuple(self.config.normalize_std),
            crop_mode="resize",
        )

    def partition_videos(
        self,
        video_entries: List[Dict[str, Union[str, int]]]
    ) -> Dict[str, List[Dict[str, Union[str, int]]]]:
        """
        Partitions raw video entries into train, validation, and test splits.
        Groups by original video stem / source_id to PREVENT data leakage across splits.
        """
        # Seed deterministic shuffle
        rng = random.Random(self.config.random_seed)

        # Group videos by label for stratified splitting
        by_label: Dict[str, List[Dict[str, Union[str, int]]]] = {
            "violence": [],
            "non_violence": [],
        }

        for item in video_entries:
            lbl = str(item.get("label", "non_violence")).lower()
            if lbl in by_label:
                by_label[lbl].append(item)
            else:
                by_label["non_violence"].append(item)

        splits: Dict[str, List[Dict[str, Union[str, int]]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for lbl, items in by_label.items():
            # Group by unique base video source to ensure clips from the same source stay together
            source_groups: Dict[str, List[Dict[str, Union[str, int]]]] = {}
            for entry in items:
                v_path = str(entry["video_path"])
                # Source ID is base filename minus clip index or parent folder
                base_name = os.path.splitext(os.path.basename(v_path))[0]
                source_id = str(entry.get("source_id", base_name))
                if source_id not in source_groups:
                    source_groups[source_id] = []
                source_groups[source_id].append(entry)

            unique_sources = list(source_groups.keys())
            rng.shuffle(unique_sources)

            n_total = len(unique_sources)
            n_train = int(math.ceil(n_total * self.config.train_ratio))
            n_val = int(math.ceil(n_total * self.config.val_ratio))

            train_srcs = set(unique_sources[:n_train])
            val_srcs = set(unique_sources[n_train:n_train + n_val])
            test_srcs = set(unique_sources[n_train + n_val:])

            for src, group in source_groups.items():
                if src in train_srcs:
                    for g in group:
                        g["split"] = "train"
                    splits["train"].extend(group)
                elif src in val_srcs:
                    for g in group:
                        g["split"] = "val"
                    splits["val"].extend(group)
                else:
                    for g in group:
                        g["split"] = "test"
                    splits["test"].extend(group)

        return splits

    def create_processed_structure(self, base_dir: Optional[str] = None) -> None:
        """Creates the directory hierarchy for processed datasets and annotations."""
        root = base_dir or self.config.processed_dir
        for split in ["train", "val", "test"]:
            for cat in ["violence", "non_violence"]:
                p = os.path.join(root, split, cat)
                os.makedirs(p, exist_ok=True)

        os.makedirs(self.config.annotations_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.raw_dir, "violence"), exist_ok=True)
        os.makedirs(os.path.join(self.config.raw_dir, "non_violence"), exist_ok=True)

    def write_manifest(
        self,
        clips: List[PreprocessedClip],
        manifest_csv_path: Optional[str] = None,
        summary_json_path: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Writes CSV and JSON dataset manifests containing metadata, labels, and split assignments.
        """
        self.create_processed_structure()
        csv_path = manifest_csv_path or os.path.join(self.config.annotations_dir, "labels.csv")
        json_path = summary_json_path or os.path.join(self.config.annotations_dir, "manifest.json")

        # 1. Write CSV
        fieldnames = [
            "clip_id", "source_video_path", "processed_clip_path", "split",
            "label", "label_id", "subcategory", "source_dataset",
            "start_frame", "end_frame", "start_sec", "end_sec", "num_frames"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in clips:
                writer.writerow(c.to_dict())

        # 2. Compute Summary Statistics
        split_counts = {"train": 0, "val": 0, "test": 0}
        label_counts = {"violence": 0, "non_violence": 0}
        subcategory_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}

        for c in clips:
            split_counts[c.split] = split_counts.get(c.split, 0) + 1
            label_counts[c.label] = label_counts.get(c.label, 0) + 1
            subcategory_counts[c.subcategory] = subcategory_counts.get(c.subcategory, 0) + 1
            source_counts[c.source_dataset] = source_counts.get(c.source_dataset, 0) + 1

        summary = {
            "total_clips": len(clips),
            "split_distribution": split_counts,
            "label_distribution": label_counts,
            "subcategory_distribution": subcategory_counts,
            "sources": source_counts,
            "spatiotemporal_format": {
                "num_frames": self.config.num_frames,
                "temporal_stride": self.config.temporal_stride,
                "target_height": self.config.target_height,
                "target_width": self.config.target_width,
                "fps_target": self.config.fps_target,
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return csv_path, json_path
