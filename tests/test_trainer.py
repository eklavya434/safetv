"""
Unit tests for SafeTV model training, parameter freezing, optimization,
checkpoint saving, and test-set evaluation.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = object

from safetv.metrics import EvaluationReport
from safetv.trainer import SafeTVTrainer, TrainingConfig


@unittest.skipIf(torch is None, "PyTorch required for trainer tests")
class TestSafeTVTrainer(unittest.TestCase):
    """Tests for model fine-tuning architecture and training mechanics."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            architecture="r2plus1d_18",
            output_dir=self.temp_dir,
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            backbone_learning_rate=1e-4,
            device="cpu",
            num_threads=2,
            pretrained_backbone=False # Use uninitialized weights for rapid unit tests
        )
        self.trainer = SafeTVTrainer(self.config)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_layer_freezing_structure(self):
        model = self.trainer.build_model()
        self.assertIsNotNone(model)

        # Early layers must be frozen
        for name, param in model.named_parameters():
            if any(k in name for k in ["stem", "layer1", "layer2"]):
                self.assertFalse(param.requires_grad, f"Layer {name} should be frozen!")
            elif "fc" in name:
                self.assertTrue(param.requires_grad, f"Classification head {name} must be trainable!")

    def test_forward_backward_step(self):
        model = self.trainer.build_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

        # Dummy batch of 2 video windows: (B=2, C=3, T=16, H=112, W=112)
        dummy_videos = torch.randn(2, 3, 16, 112, 112, dtype=torch.float32)
        dummy_labels = torch.tensor([0, 1], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(dummy_videos)
        loss = criterion(logits, dummy_labels)
        loss.backward()

        # Check gradients exist on trainable layers
        has_grads = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.fc.parameters())
        self.assertTrue(has_grads, "No gradients computed for classification head!")
        optimizer.step()

    def test_checkpoint_and_metadata_serialization(self):
        self.trainer.build_model()
        mock_report = EvaluationReport(
            total_samples=10,
            true_positives=4,
            false_positives=1,
            true_negatives=4,
            false_negatives=1,
            accuracy=0.80,
            precision=0.80,
            recall=0.80,
            specificity=0.80,
            f1_score=0.80,
            balanced_accuracy=0.80,
            false_alarm_rate=0.20
        )
        metadata_dict = {"notes": "unit test checkpoint"}

        ckpt_path = self.trainer.save_checkpoint(mock_report, metadata_dict, output_dir=self.temp_dir)

        self.assertTrue(os.path.exists(ckpt_path))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "model_config.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "class_mapping.json")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "training_metadata.json")))


if __name__ == "__main__":
    unittest.main()
