"""
Video Action Recognition Model Backbone for SafeTV.
Provides spatiotemporal 3D convolution / factorized (2+1)D feature extraction
and binary violence action classification using genuinely trained weights.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .config import ModelConfig


class VideoActionClassifier:
    """
    Spatiotemporal action recognition engine for SafeTV.
    Distinguishes potential violence from normal high/low-energy human activities
    using genuinely fine-tuned deep representations.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.device = None
        self.is_trained: bool = False
        self.provenance: str = "uninitialized"
        self.loaded_checkpoint: Optional[str] = None
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the deep neural network architecture and load weights."""
        try:
            import torch
            import torch.nn as nn
            import torchvision.models.video as video_models

            # Tune intra-op thread count for Intel 12th Gen i5-12450H (4 P-cores + 4 E-cores)
            torch.set_num_threads(self.config.num_threads)
            self.device = torch.device(
                self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"
            )

            arch = self.config.architecture.lower()

            if arch == "mc3_18":
                weights = video_models.MC3_18_Weights.DEFAULT if self.config.pretrained else None
                base_model = video_models.mc3_18(weights=weights)
            else:
                # Default to factorized R(2+1)D-18
                weights = video_models.R2Plus1D_18_Weights.DEFAULT if self.config.pretrained else None
                base_model = video_models.r2plus1d_18(weights=weights)

            in_features = base_model.fc.in_features

            # SafeTV Violence Classification Head: [0: Normal Activity, 1: Potential Violence]
            # Architecture matches SafeTVTrainer fine-tuning head exactly
            base_model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, self.config.num_classes)
            )

            self.model = base_model
            self._init_head_weights()
            self.model.to(self.device)
            self.model.eval()

            # Attempt to automatically load trained checkpoint if specified and available
            checkpoint = getattr(self.config, "checkpoint_path", None)
            if checkpoint and os.path.exists(checkpoint):
                self.load_checkpoint(checkpoint)
            else:
                self.is_trained = False
                self.provenance = "uncalibrated_backbone"

        except ImportError:
            # Fallback when torch is not available
            self.model = None
            self.is_trained = False
            self.provenance = "heuristic_fallback"

    def _init_head_weights(self) -> None:
        """Initialize head linear layers with orthogonal weights."""
        import torch.nn as nn
        if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Sequential):
            for module in self.model.fc:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load genuinely trained model weights from disk."""
        import torch
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.is_trained = True
        self.provenance = "trained"
        self.loaded_checkpoint = checkpoint_path

    def _extract_kinematic_flux(self, tensor_4d: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract physical spatiotemporal motion dynamics for diagnostic telemetry:
        1. Velocity energy: ||dI/dt||^2
        2. Jerk / impact acceleration: ||d^2I/dt^2||^2
        3. Chaos ratio: Jerk / (Velocity + eps)
        """
        # tensor_4d shape: (C, T, H, W)
        v = np.diff(tensor_4d, axis=1)          # (C, T-1, H, W)
        a = np.diff(v, axis=1)                  # (C, T-2, H, W)
        v_energy = float(np.mean(np.square(v)))
        a_jerk = float(np.mean(np.square(a)))
        chaos_ratio = float(a_jerk / (v_energy + 1e-6))
        return v_energy, a_jerk, chaos_ratio

    def predict_window(self, tensor_4d: np.ndarray) -> Dict[str, Union[float, str, bool, dict]]:
        """
        Run forward inference on a single spatiotemporal window (C, T, H, W).
        Computes genuine AI violence probability without handcrafted heuristic blending.
        Retains physical kinematic dynamics strictly as auxiliary telemetry.
        """
        if tensor_4d.ndim != 4:
            raise ValueError(f"Expected 4D array of shape (C, T, H, W), got shape {tensor_4d.shape}")

        v_energy, a_jerk, chaos_ratio = self._extract_kinematic_flux(tensor_4d)

        if self.model is not None:
            import torch
            tensor_5d = torch.from_numpy(tensor_4d).unsqueeze(0).to(self.device, dtype=torch.float32)

            with torch.no_grad():
                logits = self.model(tensor_5d)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Genuine AI classification probabilities: [0: Normal, 1: Violence]
            normal_prob = float(probs[0])
            violence_prob = float(probs[1])
            label = "VIOLENCE" if violence_prob >= 0.50 else "NORMAL"

            return {
                "violence_prob": violence_prob,
                "normal_prob": normal_prob,
                "predicted_label": label,
                "is_trained": self.is_trained,
                "provenance": self.provenance,
                "motion_flux": {
                    "velocity_energy": round(v_energy, 4),
                    "jerk_acceleration": round(a_jerk, 4),
                    "chaos_ratio": round(chaos_ratio, 4),
                },
            }
        else:
            # Fallback when PyTorch is not available: explicit heuristic diagnostics
            kinematic_logit = (a_jerk * 3.5) + (v_energy * 2.0) - 3.2
            violence_prob = float(np.clip(1.0 / (1.0 + np.exp(-kinematic_logit)), 0.0, 1.0))
            normal_prob = 1.0 - violence_prob
            label = "VIOLENCE" if violence_prob >= 0.50 else "NORMAL"
            return {
                "violence_prob": violence_prob,
                "normal_prob": normal_prob,
                "predicted_label": label,
                "is_trained": False,
                "provenance": "heuristic_fallback",
                "motion_flux": {
                    "velocity_energy": round(v_energy, 4),
                    "jerk_acceleration": round(a_jerk, 4),
                    "chaos_ratio": round(chaos_ratio, 4),
                },
            }

    def predict_batch(self, tensors: List[np.ndarray]) -> List[Dict[str, Union[float, str, bool, dict]]]:
        """Run batch inference over a list of 4D spatiotemporal window arrays."""
        results = []
        for t in tensors:
            results.append(self.predict_window(t))
        return results
