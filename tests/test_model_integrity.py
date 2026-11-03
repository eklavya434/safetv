"""
Tests for model checkpoint integrity, manifest validation, and fail-closed behavior.
"""
import hashlib
import json
import os
import shutil
import tempfile
import unittest

from safetv.model import ModelNotReadyError, VideoActionClassifier
from safetv.config import ModelConfig


class TestModelIntegrity(unittest.TestCase):
    """Test model checkpoint loading and integrity validation."""

    def setUp(self):
        self.config = ModelConfig()
        self.checkpoint_path = "models/violence_detector/best_model.pth"
        self.manifest_path = "models/violence_detector/manifest.json"

    def test_checkpoint_file_exists(self):
        """Checkpoint file must exist at configured path."""
        self.assertTrue(
            os.path.exists(self.checkpoint_path),
            f"Checkpoint not found at {self.checkpoint_path}"
        )

    def test_checkpoint_is_nonzero(self):
        """Checkpoint file must not be empty."""
        size = os.path.getsize(self.checkpoint_path)
        self.assertGreater(size, 1024, "Checkpoint file is suspiciously small")

    def test_model_loads_successfully(self):
        """Model must initialize and load checkpoint without error."""
        model = VideoActionClassifier(self.config)
        self.assertTrue(model.is_ready, f"Model not ready: {model.model_status_message}")
        self.assertTrue(model.is_trained)

    def test_class_mapping_is_correct(self):
        """Class mapping must have 0=non_violence/normal and 1=violence."""
        model = VideoActionClassifier(self.config)
        self.assertIn(0, model.class_mapping)
        self.assertIn(1, model.class_mapping)
        self.assertIn(model.class_mapping[1].lower(), ["violence"])
        self.assertIn(model.class_mapping[0].lower(), ["non_violence", "nonviolence", "normal"])

    def test_checkpoint_hash_is_set(self):
        """SHA-256 hash must be computed after loading checkpoint."""
        model = VideoActionClassifier(self.config)
        self.assertIsNotNone(model.checkpoint_hash)
        self.assertEqual(len(model.checkpoint_hash), 64)

    def test_predict_window_raises_when_not_ready(self):
        """predict_window must raise ModelNotReadyError when is_ready=False."""
        import numpy as np
        model = VideoActionClassifier(self.config)
        model.is_ready = False
        dummy = np.random.rand(3, 16, 112, 112).astype("float32")
        with self.assertRaises(ModelNotReadyError):
            model.predict_window(dummy, allow_uncalibrated=False)

    def test_model_weights_are_finite(self):
        """All model parameters must be finite (no NaN/Inf)."""
        import torch
        model = VideoActionClassifier(self.config)
        self.assertTrue(model.is_ready)
        for name, param in model.model.named_parameters():
            self.assertTrue(
                torch.isfinite(param).all().item(),
                f"Non-finite weights in parameter: {name}"
            )

    def test_manifest_exists_and_valid(self):
        """Manifest JSON must exist and contain required fields."""
        self.assertTrue(os.path.exists(self.manifest_path), "manifest.json not found")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        required_fields = ["architecture", "num_classes", "class_mapping", "checkpoint_sha256", "is_production_ready"]
        for field in required_fields:
            self.assertIn(field, manifest, f"Missing field '{field}' in manifest.json")
        self.assertFalse(manifest["is_production_ready"], "is_production_ready must be False for current prototype")

    def test_manifest_hash_matches_checkpoint(self):
        """Manifest SHA-256 must match actual checkpoint file hash."""
        self.assertTrue(os.path.exists(self.manifest_path), "manifest.json not found")
        sha256 = hashlib.sha256()
        with open(self.checkpoint_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest().lower()
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        expected = manifest.get("checkpoint_sha256", "").lower()
        self.assertEqual(actual_hash, expected, "Manifest SHA-256 does not match checkpoint file")

    def test_corrupted_checkpoint_byte_flip_refused(self):
        """Flipping a single byte in the checkpoint must cause SHA-256 validation failure and reject loading."""
        temp_dir = tempfile.mkdtemp()
        try:
            temp_ckpt = os.path.join(temp_dir, "best_model.pth")
            temp_manifest = os.path.join(temp_dir, "manifest.json")
            shutil.copyfile(self.checkpoint_path, temp_ckpt)
            shutil.copyfile(self.manifest_path, temp_manifest)

            # Flip one byte in the checkpoint copy
            with open(temp_ckpt, "r+b") as f:
                f.seek(100)
                b = f.read(1)
                f.seek(100)
                f.write(bytes([(b[0] ^ 0xFF)]))

            cfg = ModelConfig(checkpoint_path=temp_ckpt)
            model = VideoActionClassifier(cfg)
            with self.assertRaises(ModelNotReadyError):
                model.load_checkpoint(temp_ckpt)
            self.assertFalse(model.is_ready)
            self.assertEqual(model.provenance, "integrity_failure")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_missing_manifest_refused(self):
        """Loading a checkpoint without a companion manifest.json must be refused."""
        temp_dir = tempfile.mkdtemp()
        try:
            temp_ckpt = os.path.join(temp_dir, "best_model.pth")
            shutil.copyfile(self.checkpoint_path, temp_ckpt)
            # Do NOT copy manifest.json

            cfg = ModelConfig(checkpoint_path=temp_ckpt)
            model = VideoActionClassifier(cfg)
            with self.assertRaises(ModelNotReadyError):
                model.load_checkpoint(temp_ckpt)
            self.assertFalse(model.is_ready)
            self.assertEqual(model.provenance, "missing_manifest")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_nan_weights_refused(self):
        """A checkpoint containing non-finite (NaN/Inf) weights must be rejected."""
        import torch
        temp_dir = tempfile.mkdtemp()
        try:
            temp_ckpt = os.path.join(temp_dir, "best_model.pth")
            temp_manifest = os.path.join(temp_dir, "manifest.json")

            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            first_key = next(iter(state.keys()))
            state[first_key][0] = float("nan")
            torch.save(state, temp_ckpt)

            sha256 = hashlib.sha256()
            with open(temp_ckpt, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha256.update(chunk)

            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest_data["checkpoint_sha256"] = sha256.hexdigest()
            with open(temp_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f)

            cfg = ModelConfig(checkpoint_path=temp_ckpt)
            model = VideoActionClassifier(cfg)
            with self.assertRaises(ModelNotReadyError):
                model.load_checkpoint(temp_ckpt)
            self.assertFalse(model.is_ready)
            self.assertEqual(model.provenance, "corrupted_checkpoint")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
