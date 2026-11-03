"""
PyTorch Dataset and DataLoader implementation for SafeTV.
Loads standardized spatiotemporal video windows with on-the-fly augmentation and metadata.
"""

import csv
import os
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None

from .config import DatasetConfig
from .preprocessor import VideoPreprocessor


class SafeTVVideoDataset(Dataset):
    """
    PyTorch Video Dataset for physical violence vs normal human activity classification.
    """

    def __init__(
        self,
        manifest_csv_path: str,
        split: Optional[str] = None,
        config: Optional[DatasetConfig] = None,
        transform: Optional[Callable] = None,
        is_training: bool = False
    ):
        self.config = config or DatasetConfig()
        self.manifest_csv_path = manifest_csv_path
        self.split = split
        self.transform = transform
        self.is_training = is_training
        self.preprocessor = VideoPreprocessor(self.config)

        self.entries: List[Dict[str, str]] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Reads annotations from manifest CSV file, filtering by split if specified."""
        if not os.path.exists(self.manifest_csv_path):
            raise FileNotFoundError(f"Manifest CSV not found: {self.manifest_csv_path}")

        with open(self.manifest_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.split is None or row.get("split", "").lower() == self.split.lower():
                    self.entries.append(row)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Union["torch.Tensor", str, int]]:
        if torch is None:
            raise RuntimeError("PyTorch is required to use SafeTVVideoDataset.")

        entry = self.entries[idx]
        v_path = entry.get("processed_clip_path") or entry.get("source_video_path", "")
        start_f = int(entry.get("start_frame", 0))

        # Random temporal jitter during training
        if self.is_training:
            jitter = random.randint(-4, 4)
            start_f = max(0, start_f + jitter)

        # Extract normalized 4D spatiotemporal array (C, T, H, W)
        tensor_np = self.preprocessor.extract_clip_tensor(v_path, start_frame=start_f)

        if tensor_np is None:
            raise RuntimeError(
                f"Failed to decode video frames for clip '{entry.get('clip_id', idx)}' at path '{v_path}'. "
                "Corrupted frames cannot be silently replaced with zero tensors."
            )

        # Training data augmentation
        if self.is_training:
            # Random horizontal spatial flip
            if random.random() > 0.5:
                tensor_np = np.flip(tensor_np, axis=3).copy() # Flip W dimension

        if self.transform is not None:
            tensor_np = self.transform(tensor_np)

        video_tensor = torch.from_numpy(tensor_np).float()
        label_id = int(entry.get("label_id", 0))
        label_tensor = torch.tensor(label_id, dtype=torch.long)

        return {
            "video": video_tensor,
            "label": label_tensor,
            "clip_id": entry.get("clip_id", f"clip_{idx}"),
            "subcategory": entry.get("subcategory", "unknown"),
            "source_dataset": entry.get("source_dataset", "unknown"),
        }


def create_video_dataloaders(
    manifest_csv_path: str,
    batch_size: int = 4,
    num_workers: int = 0,
    config: Optional[DatasetConfig] = None
) -> Dict[str, "DataLoader"]:
    """
    Factory function to construct train, validation, and test PyTorch DataLoaders.
    """
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch is required to create DataLoaders.")

    cfg = config or DatasetConfig()

    train_ds = SafeTVVideoDataset(manifest_csv_path, split="train", config=cfg, is_training=True)
    val_ds = SafeTVVideoDataset(manifest_csv_path, split="val", config=cfg, is_training=False)
    test_ds = SafeTVVideoDataset(manifest_csv_path, split="test", config=cfg, is_training=False)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=len(train_ds) > batch_size
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        ),
    }

    return loaders
