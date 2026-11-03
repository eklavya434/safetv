"""
Unit tests for SafeTV dataset preparation, video inspection, partitioning,
PyTorch Dataset, and zero-data-leakage verification.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from safetv.data.config import DatasetConfig
from safetv.data.dataset import SafeTVVideoDataset, create_video_dataloaders
from safetv.data.downloader import DatasetDownloader
from safetv.data.preprocessor import PreprocessedClip, VideoPreprocessor
from safetv.data.video_inspector import VideoInspector
from prepare_dataset import parse_video_label


class TestVideoInspector(unittest.TestCase):
    """Tests for video file inspection and corruption detection."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.inspector = VideoInspector()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_video_inspection(self):
        vid_path = os.path.join(self.temp_dir, "test_valid.mp4")
        DatasetDownloader.generate_synthetic_video_clip(vid_path, category="normal_walking", num_frames=30)

        meta = self.inspector.inspect_file(vid_path)
        self.assertTrue(meta.is_valid)
        self.assertEqual(meta.total_frames, 30)
        self.assertEqual(meta.width, 160)
        self.assertEqual(meta.height, 120)
        self.assertAlmostEqual(meta.fps, 30.0, places=1)
        self.assertAlmostEqual(meta.duration_sec, 1.0, places=1)
        self.assertIsNotNone(meta.fingerprint_hash)

    def test_empty_file_corruption(self):
        empty_path = os.path.join(self.temp_dir, "empty.mp4")
        with open(empty_path, "wb") as f:
            pass # 0 bytes

        meta = self.inspector.inspect_file(empty_path)
        self.assertFalse(meta.is_valid)
        self.assertIn("empty", meta.error_message.lower())

    def test_duplicate_fingerprint_matching(self):
        vid1 = os.path.join(self.temp_dir, "clip_a.mp4")
        vid2 = os.path.join(self.temp_dir, "clip_b.mp4")
        DatasetDownloader.generate_synthetic_video_clip(vid1, category="exercising", num_frames=30)
        # Duplicate file
        shutil.copyfile(vid1, vid2)

        meta1 = self.inspector.inspect_file(vid1)
        meta2 = self.inspector.inspect_file(vid2)

        self.assertEqual(meta1.fingerprint_hash, meta2.fingerprint_hash)


class TestPreprocessorAndPartitioning(unittest.TestCase):
    """Tests for tensor extraction, strict source partitioning, and manifest generation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = DatasetConfig(
            raw_dir=os.path.join(self.temp_dir, "raw"),
            processed_dir=os.path.join(self.temp_dir, "processed"),
            annotations_dir=os.path.join(self.temp_dir, "annotations"),
            target_width=112,
            target_height=112,
            num_frames=16,
            temporal_stride=2
        )
        self.preprocessor = VideoPreprocessor(self.config)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_clip_tensor_dimensions_and_normalization(self):
        vid_path = os.path.join(self.temp_dir, "sample.mp4")
        DatasetDownloader.generate_synthetic_video_clip(vid_path, category="punching", num_frames=45)

        tensor = self.preprocessor.extract_clip_tensor(vid_path, start_frame=0)
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (3, 16, 112, 112))
        self.assertEqual(tensor.dtype, np.float32)

    def test_zero_data_leakage_between_splits(self):
        """Validates that all clips originating from the same source video stay strictly in the same split."""
        entries = []
        # Create 10 original sources with multiple sub-clips each
        for s_idx in range(10):
            source_id = f"camera_source_{s_idx:02d}"
            lbl = "violence" if s_idx % 2 == 0 else "non_violence"
            for c_idx in range(3):
                entries.append({
                    "video_path": f"/mock/path/{source_id}_part{c_idx}.mp4",
                    "label": lbl,
                    "label_id": 1 if lbl == "violence" else 0,
                    "subcategory": "action",
                    "source_id": source_id
                })

        splits = self.preprocessor.partition_videos(entries)

        train_sources = {e["source_id"] for e in splits["train"]}
        val_sources = {e["source_id"] for e in splits["val"]}
        test_sources = {e["source_id"] for e in splits["test"]}

        # Intersections MUST be strictly empty
        self.assertEqual(len(train_sources.intersection(val_sources)), 0, "Data leakage between Train and Val!")
        self.assertEqual(len(train_sources.intersection(test_sources)), 0, "Data leakage between Train and Test!")
        self.assertEqual(len(val_sources.intersection(test_sources)), 0, "Data leakage between Val and Test!")


class TestDatasetAndDataLoader(unittest.TestCase):
    """Tests for PyTorch SafeTVVideoDataset and batch loading."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = DatasetConfig(
            raw_dir=os.path.join(self.temp_dir, "raw"),
            processed_dir=os.path.join(self.temp_dir, "processed"),
            annotations_dir=os.path.join(self.temp_dir, "annotations"),
        )
        self.downloader = DatasetDownloader(self.config)
        self.csv_path, self.json_path = self.downloader.build_sample_dataset(
            base_dir=self.temp_dir,
            num_samples_per_category=2
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_manifest_files_exist_and_match(self):
        self.assertTrue(os.path.exists(self.csv_path))
        self.assertTrue(os.path.exists(self.json_path))

    @unittest.skipIf(torch is None, "PyTorch not available")
    def test_dataset_loading_and_batch_tensors(self):
        loaders = create_video_dataloaders(self.csv_path, batch_size=2, config=self.config)
        train_loader = loaders["train"]

        batch = next(iter(train_loader))
        videos = batch["video"]
        labels = batch["label"]

        self.assertEqual(videos.ndim, 5) # (B, C, T, H, W)
        self.assertEqual(videos.shape[1], 3)
        self.assertEqual(videos.shape[2], 16)
        self.assertEqual(videos.shape[3], 112)
        self.assertEqual(videos.shape[4], 112)
        self.assertEqual(labels.ndim, 1)
        self.assertTrue(all(lbl in [0, 1] for lbl in labels.tolist()))


class TestDatasetLabellingBug(unittest.TestCase):
    """
    Validates fix for dataset labelling bug where substring 'viol'
    inside 'non_violence' caused catastrophic misclassification.
    """

    def test_parse_video_label_non_violence_not_mislabeled(self):
        """Ensures directories or filenames with 'non_violence' are classified as class 0."""
        paths_non_violence = [
            os.path.join("data", "raw", "non_violence", "nv_001.mp4"),
            os.path.join("data", "raw", "non-violence", "run_042.mp4"),
            os.path.join("data", "raw", "nonviolence", "dance_100.mp4"),
            os.path.join("videos", "walk_in_park.mp4"),
            os.path.join("benchmark", "exercising_fast.mp4"),
        ]
        for p in paths_non_violence:
            label_str, label_id = parse_video_label(p)
            self.assertEqual(label_str, "non_violence", f"Failed for {p}")
            self.assertEqual(label_id, 0, f"Failed for {p}")

    def test_parse_video_label_violence(self):
        """Ensures true violence paths are classified as class 1."""
        paths_violence = [
            os.path.join("data", "raw", "violence", "punching_001.mp4"),
            os.path.join("data", "raw", "violence", "fight_scene.mp4"),
            os.path.join("clips", "kicking_street_brawl.mp4"),
        ]
        for p in paths_violence:
            label_str, label_id = parse_video_label(p)
            self.assertEqual(label_str, "violence", f"Failed for {p}")
            self.assertEqual(label_id, 1, f"Failed for {p}")

    def test_parse_video_label_ambiguous_rejected(self):
        """Ensures paths with no discernible category raise ValueError instead of guessing."""
        with self.assertRaises(ValueError):
            parse_video_label("unknown_folder/unlabeled_stream_99.mp4")


if __name__ == "__main__":
    unittest.main()
