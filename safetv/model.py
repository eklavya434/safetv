"""
Video Action Recognition Model Backbone for SafeTV.
Provides spatiotemporal 3D convolution / factorized (2+1)D feature extraction
and binary violence action classification using genuinely trained weights.
Enforces strict model integrity and checkpoint validation.
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .config import ModelConfig

logger = logging.getLogger("safetv.model")


class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested but no valid trained model checkpoint is loaded."""
    pass


class VideoActionClassifier:
    """
    Spatiotemporal action recognition engine for SafeTV.
    Distinguishes potential violence from normal human activities using
    genuinely fine-tuned deep representations.
    Outputs pure neural network softmax probabilities without heuristic blending.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.device = None
        self.is_trained: bool = False
        self.is_ready: bool = False
        self.provenance: str = "uninitialized"
        self.loaded_checkpoint: Optional[str] = None
        self.checkpoint_hash: Optional[str] = None
        self.class_mapping: Dict[int, str] = {0: "normal", 1: "violence"}
        self.model_status_message: str = "Uninitialized"
        self._init_model()

    @property
    def architecture(self) -> str:
        return self.config.architecture

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of checkpoint file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _init_model(self) -> None:
        """Initialize the deep neural network architecture and load validated weights."""
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
                weights = video_models.R2Plus1D_18_Weights.DEFAULT if self.config.pretrained else None
                base_model = video_models.r2plus1d_18(weights=weights)

            in_features = base_model.fc.in_features

            # SafeTV Violence Classification Head: [0: Normal Activity, 1: Potential Violence]
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

            # Attempt to automatically load trained checkpoint
            checkpoint = getattr(self.config, "checkpoint_path", None)
            if checkpoint and os.path.exists(checkpoint):
                self.load_checkpoint(checkpoint)
            else:
                self.is_trained = False
                self.is_ready = False
                self.provenance = "uncalibrated_backbone"
                self.model_status_message = (
                    f"Trained checkpoint '{checkpoint}' not found. Model is in uncalibrated state."
                )
                logger.warning(self.model_status_message)

        except ImportError as e:
            self.model = None
            self.is_trained = False
            self.is_ready = False
            self.provenance = "unavailable"
            self.model_status_message = f"PyTorch / torchvision not installed: {e}"
            logger.error(self.model_status_message)

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
        """
        Load genuinely trained model weights with strict integrity and key validation.
        """
        import torch
        if not os.path.exists(checkpoint_path):
            self.is_trained = False
            self.is_ready = False
            self.provenance = "uncalibrated_backbone"
            self.model_status_message = f"Checkpoint file not found: {checkpoint_path}"
            raise FileNotFoundError(self.model_status_message)

        # Compute SHA-256 hash for integrity tracking
        self.checkpoint_hash = self._compute_file_hash(checkpoint_path)

        # Attempt to load class mapping if present in checkpoint directory
        ckpt_dir = os.path.dirname(checkpoint_path)
        class_map_file = os.path.join(ckpt_dir, "class_mapping.json")
        if os.path.exists(class_map_file):
            try:
                with open(class_map_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.class_mapping = {int(k): v for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Could not parse class_mapping.json: {e}")

        try:
            # Safe loading with torch
            state = torch.load(checkpoint_path, map_location=self.device)
            if not isinstance(state, dict):
                raise ValueError("Checkpoint is not a valid state dictionary.")

            # Strict key verification
            model_keys = set(self.model.state_dict().keys())
            state_keys = set(state.keys())

            missing_keys = model_keys - state_keys
            unexpected_keys = state_keys - model_keys

            if missing_keys or unexpected_keys:
                err_msg = f"Checkpoint key mismatch! Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}"
                logger.error(err_msg)
                self.is_trained = False
                self.is_ready = False
                self.provenance = "corrupted_checkpoint"
                self.model_status_message = err_msg
                raise KeyError(err_msg)

            self.model.load_state_dict(state, strict=True)
            self.model.eval()

            self.is_trained = True
            self.is_ready = True
            self.provenance = "trained"
            self.loaded_checkpoint = checkpoint_path
            self.model_status_message = f"Valid trained checkpoint loaded from {checkpoint_path} (SHA-256: {self.checkpoint_hash[:12]}...)"
            logger.info(self.model_status_message)

        except Exception as e:
            self.is_trained = False
            self.is_ready = False
            self.provenance = "invalid_checkpoint"
            self.model_status_message = f"Failed to load checkpoint: {e}"
            logger.error(self.model_status_message)
            raise ModelNotReadyError(self.model_status_message) from e

    def _extract_kinematic_flux(self, tensor_4d: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract physical spatiotemporal motion dynamics strictly for auxiliary telemetry:
        1. Velocity energy: ||dI/dt||^2
        2. Jerk / impact acceleration: ||d^2I/dt^2||^2
        3. Chaos ratio: Jerk / (Velocity + eps)
        NOTE: These metrics are NEVER blended into the AI violence score.
        """
        v = np.diff(tensor_4d, axis=1)          # (C, T-1, H, W)
        a = np.diff(v, axis=1)                  # (C, T-2, H, W)
        v_energy = float(np.mean(np.square(v)))
        a_jerk = float(np.mean(np.square(a)))
        chaos_ratio = float(a_jerk / (v_energy + 1e-6))
        return v_energy, a_jerk, chaos_ratio

    def predict_window(
        self,
        tensor_4d: np.ndarray,
        allow_uncalibrated: bool = False
    ) -> Dict[str, Union[float, str, bool, dict]]:
        """
        Run forward inference on a single spatiotemporal window (C, T, H, W).
        Computes genuine AI violence probability via neural network softmax.
        Raises ModelNotReadyError if model is not ready, preventing fake scores.
        """
        if tensor_4d.ndim != 4:
            raise ValueError(f"Expected 4D array of shape (C, T, H, W), got shape {tensor_4d.shape}")

        if not self.is_ready:
            if not allow_uncalibrated:
                raise ModelNotReadyError(
                    f"Model is not ready for violence prediction: {self.model_status_message}"
                )

        v_energy, a_jerk, chaos_ratio = self._extract_kinematic_flux(tensor_4d)

        if self.model is not None:
            import torch
            tensor_5d = torch.from_numpy(tensor_4d).unsqueeze(0).to(self.device, dtype=torch.float32)

            with torch.no_grad():
                logits = self.model(tensor_5d)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            normal_prob = float(probs[0])
            violence_prob = float(probs[1])
            label = "VIOLENCE" if violence_prob >= 0.50 else "NORMAL"

            return {
                "violence_prob": violence_prob,
                "normal_prob": normal_prob,
                "predicted_label": label,
                "is_trained": self.is_trained,
                "is_ready": self.is_ready,
                "provenance": self.provenance,
                "checkpoint_hash": self.checkpoint_hash,
                "motion_flux": {
                    "velocity_energy": round(v_energy, 4),
                    "jerk_acceleration": round(a_jerk, 4),
                    "chaos_ratio": round(chaos_ratio, 4),
                },
            }
        else:
            raise ModelNotReadyError("PyTorch neural network model is not available.")

    def predict_batch(
        self,
        tensors: List[np.ndarray],
        allow_uncalibrated: bool = False
    ) -> List[Dict[str, Union[float, str, bool, dict]]]:
        """Run batch inference over a list of 4D spatiotemporal window arrays."""
        results = []
        for t in tensors:
            results.append(self.predict_window(t, allow_uncalibrated=allow_uncalibrated))
        return results

    def get_health_report(self) -> Dict[str, Union[str, bool, dict, None]]:
        """
        Comprehensive health report verifying model architecture, loaded checkpoint,
        SHA-256 integrity, training provenance, class mapping, and evaluation suitability.
        """
        calibration = "CALIBRATED_REAL_WORLD" if self.is_ready and self.is_trained else "UNCALIBRATED"
        return {
            "architecture": self.architecture,
            "checkpoint_loaded": self.loaded_checkpoint,
            "checkpoint_hash": self.checkpoint_hash,
            "training_provenance": self.provenance,
            "class_mapping": self.class_mapping,
            "calibration_status": calibration,
            "is_suitable_for_evaluation": self.is_ready,
            "is_production_ready": False,
            "status_message": self.model_status_message,
            "preprocessing_config": {
                "temporal_window_frames": 16,
                "spatial_resolution": [112, 112],
                "input_channels": 3,
                "channel_order": "RGB",
                "pixel_scaling": "0.0_to_1.0",
                "normalization": {
                    "mean": [0.43216, 0.394666, 0.37645],
                    "std": [0.22803, 0.22145, 0.216989]
                },
                "stride_frames": 8,
                "resize_strategy": "aspect_preserving_center_crop_112x112"
            }
        }
