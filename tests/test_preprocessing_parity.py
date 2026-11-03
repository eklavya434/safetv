"""
Tests verifying mathematical parity between training and inference preprocessing.
Ensures that VideoWindowExtractor (inference) and VideoPreprocessor (training)
produce identical normalized (C, T, H, W) tensors.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from safetv.data.config import DatasetConfig
from safetv.data.downloader import DatasetDownloader
from safetv.data.preprocessor import VideoPreprocessor
from safetv.video_loader import VideoExtractionConfig, VideoWindowExtractor
from safetv.preprocessing import standardize_frame_sequence, KINETICS_MEAN, KINETICS_STD


class TestPreprocessingParity(unittest.TestCase):
    """Verifies that inference and training preprocessing pipelines are identical."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "parity_test.mp4")
        # Generate 45 frames of synthetic video
        DatasetDownloader.generate_synthetic_video_clip(
            self.video_path,
            category="punching",
            num_frames=45,
            width=160,
            height=120,
            fps=30
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_preprocessing_function_parity(self):
        """Directly tests standardize_frame_sequence with synthetic frames."""
        rng = np.random.RandomState(42)
        raw_frames = [rng.randint(0, 256, (120, 160, 3), dtype=np.uint8) for _ in range(16)]

        tensor_a = standardize_frame_sequence(raw_frames, target_hw=(112, 112), crop_mode="resize")
        tensor_b = standardize_frame_sequence(raw_frames, target_hw=(112, 112), crop_mode="resize")

        self.assertEqual(tensor_a.shape, (3, 16, 112, 112))
        self.assertEqual(tensor_a.dtype, np.float32)
        np.testing.assert_allclose(tensor_a, tensor_b, atol=1e-6)

    def test_pipeline_components_parity_on_file(self):
        """
        Extracts the first 16-frame window using VideoPreprocessor (training)
        and VideoWindowExtractor (inference) on the exact same video file.
        """
        # Training preprocessor
        dataset_cfg = DatasetConfig(
            num_frames=16,
            temporal_stride=2,
            target_width=112,
            target_height=112,
        )
        training_preproc = VideoPreprocessor(dataset_cfg)
        train_tensor = training_preproc.extract_clip_tensor(self.video_path, start_frame=0, temporal_stride=2)

        self.assertIsNotNone(train_tensor)
        self.assertEqual(train_tensor.shape, (3, 16, 112, 112))

        # Inference extractor
        infer_cfg = VideoExtractionConfig(
            window_size=16,
            temporal_stride=2,
            window_step=16,
            target_width=112,
            target_height=112,
        )
        infer_extractor = VideoWindowExtractor(infer_cfg)
        
        # Read frames from the video file in sequence
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        windows = list(infer_extractor.extract_from_frames(all_frames, fps=30.0))
        self.assertGreater(len(windows), 0)
        infer_tensor = windows[0].tensor

        self.assertEqual(infer_tensor.shape, (3, 16, 112, 112))
        np.testing.assert_allclose(train_tensor, infer_tensor, atol=1e-5)

    def test_fail_closed_on_corrupt_frames(self):
        """Ensures corrupted/empty inputs raise or return None rather than silent zero filling."""
        preproc = VideoPreprocessor()
        # Non-existent file
        res = preproc.extract_clip_tensor(os.path.join(self.temp_dir, "nonexistent.mp4"))
        self.assertIsNone(res)

        # Empty frame list
        with self.assertRaises(ValueError):
            standardize_frame_sequence([])


if __name__ == "__main__":
    unittest.main()
