"""
SafeTV Model Trainer and Evaluator.
Implements fine-tuning for R(2+1)D-18 / MC3-18 backbones, learning rate scheduling,
early stopping, checkpoint management, and honest test-set evaluation.
"""

import datetime
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    import torchvision.models.video as video_models
except ImportError:
    torch = None
    nn = object
    AdamW = None
    CosineAnnealingLR = None
    video_models = None

from safetv.config import ModelConfig
from safetv.data.config import DatasetConfig
from safetv.data.dataset import SafeTVVideoDataset, create_video_dataloaders
from safetv.metrics import EvaluationReport, ViolenceDetectionEvaluator


@dataclass
class TrainingConfig:
    """Hyperparameters and execution configuration for model training."""
    architecture: str = "r2plus1d_18"
    pretrained_backbone: bool = True
    manifest_csv: str = "data/annotations/labels.csv"
    output_dir: str = "models/violence_detector"

    # Optimization
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4          # Learning rate for classification head
    backbone_learning_rate: float = 2e-5 # Lower learning rate for fine-tuning layer3 & layer4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 4
    min_delta: float = 0.005

    # Hardware & Threads
    device: str = "cpu"                  # "cpu" or "cuda"
    num_threads: int = 6                 # CPU thread pool for i5-12450H P-cores
    num_workers: int = 0
    random_seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


class SafeTVTrainer:
    """
    Manages end-to-end model training, validation, checkpointing, and evaluation.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self._set_seed()
        self.device = self._get_device()
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def _set_seed(self) -> None:
        """Enforces deterministic random seed across PyTorch and NumPy."""
        seed = self.config.random_seed
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _get_device(self) -> "torch.device":
        """Resolves target compute device."""
        if torch is None:
            return None
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        torch.set_num_threads(self.config.num_threads)
        return torch.device("cpu")

    def build_model(self) -> nn.Module:
        """
        Builds R(2+1)D-18 model with fine-tuning architecture:
        - Frozen: stem, layer1, layer2 (preserves low-level motion filters)
        - Trainable backbone: layer3, layer4 (adapts spatiotemporal semantics)
        - Trainable head: binary violence classifier
        """
        if torch is None or video_models is None:
            raise RuntimeError("PyTorch and torchvision are required.")

        arch = self.config.architecture.lower()
        if arch == "mc3_18":
            weights = video_models.MC3_18_Weights.DEFAULT if self.config.pretrained_backbone else None
            model = video_models.mc3_18(weights=weights)
        else:
            weights = video_models.R2Plus1D_18_Weights.DEFAULT if self.config.pretrained_backbone else None
            model = video_models.r2plus1d_18(weights=weights)

        # Freeze early layers
        for name, param in model.named_parameters():
            if any(k in name for k in ["stem", "layer1", "layer2"]):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # Replace final classification head with SafeTV Head: [0: Normal, 1: Violence]
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )

        model.to(self.device)
        self.model = model
        return model

    def setup_optimization(self, train_loader) -> None:
        """Configures parameter groups, loss function with class weighting, optimizer, and scheduler."""
        # 1. Parameter groups with differential learning rates
        head_params = []
        backbone_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": head_params, "lr": self.config.learning_rate},
            {"params": backbone_params, "lr": self.config.backbone_learning_rate},
        ]

        self.optimizer = AdamW(param_groups, weight_decay=self.config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=1e-6)

        # 2. Compute class weighting from training manifest if available
        violence_count = 0
        normal_count = 0
        for entry in getattr(train_loader.dataset, "entries", []):
            if int(entry.get("label_id", 0)) == 1:
                violence_count += 1
            else:
                normal_count += 1

        if violence_count > 0 and normal_count > 0:
            weight_for_violence = normal_count / violence_count
            class_weights = torch.tensor([1.0, float(weight_for_violence)], device=self.device)
        else:
            class_weights = torch.tensor([1.0, 1.0], device=self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for batch in train_loader:
            videos = batch["video"].to(self.device) # (B, C, T, H, W)
            labels = batch["label"].to(self.device) # (B,)

            self.optimizer.zero_grad()
            logits = self.model(videos)             # (B, 2)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.gradient_clip_norm
            )

            self.optimizer.step()

            total_loss += float(loss.item()) * videos.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total_samples += videos.size(0)

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct / max(1, total_samples)
        return epoch_loss, epoch_acc

    def evaluate(self, data_loader) -> Tuple[float, EvaluationReport, Dict[str, Dict[str, int]]]:
        """Runs evaluation over a dataset split, returning loss, metrics, and subcategory breakdown."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_labels = []
        all_preds = []
        all_probs = []

        # Subcategory error tracker
        subcategory_perf: Dict[str, Dict[str, int]] = {}

        with torch.no_grad():
            for batch in data_loader:
                videos = batch["video"].to(self.device)
                labels = batch["label"].to(self.device)
                subcats = batch.get("subcategory", ["unknown"] * videos.size(0))

                logits = self.model(videos)
                loss = self.criterion(logits, labels)
                total_loss += float(loss.item()) * videos.size(0)

                probs = torch.softmax(logits, dim=1)[:, 1] # P(violence)
                preds = torch.argmax(logits, dim=1)

                labels_list = labels.cpu().tolist()
                preds_list = preds.cpu().tolist()
                probs_list = probs.cpu().tolist()

                all_labels.extend(labels_list)
                all_preds.extend(preds_list)
                all_probs.extend(probs_list)
                total_samples += videos.size(0)

                for y_true, y_pred, subcat in zip(labels_list, preds_list, subcats):
                    if subcat not in subcategory_perf:
                        subcategory_perf[subcat] = {"correct": 0, "incorrect": 0, "total": 0}
                    subcategory_perf[subcat]["total"] += 1
                    if y_true == y_pred:
                        subcategory_perf[subcat]["correct"] += 1
                    else:
                        subcategory_perf[subcat]["incorrect"] += 1

        val_loss = total_loss / max(1, total_samples)
        metrics = ViolenceDetectionEvaluator.compute_window_metrics(all_labels, all_preds, all_probs)
        return val_loss, metrics, subcategory_perf

    def save_checkpoint(
        self,
        val_metrics: EvaluationReport,
        train_metadata: dict,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Saves best_model.pth, model_config.json, class_mapping.json, and training_metadata.json.
        """
        out_dir = output_dir or self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)

        model_path = os.path.join(out_dir, "best_model.pth")
        torch.save(self.model.state_dict(), model_path)

        # 1. model_config.json
        model_config = {
            "architecture": self.config.architecture,
            "num_classes": 2,
            "input_shape": [3, 16, 112, 112],
            "normalization_mean": [0.43216, 0.394666, 0.37645],
            "normalization_std": [0.22803, 0.22145, 0.216989],
            "num_threads": self.config.num_threads,
        }
        with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)

        # 2. class_mapping.json
        class_mapping = {
            "0": "non_violence",
            "1": "violence"
        }
        with open(os.path.join(out_dir, "class_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, indent=2)

        # 3. training_metadata.json
        metadata = {
            "model_version": "1.0.0",
            "training_date": datetime.datetime.now().isoformat(),
            "architecture": self.config.architecture,
            "training_parameters": self.config.to_dict(),
            "best_val_metrics": val_metrics.to_dict(),
            **train_metadata,
        }
        with open(os.path.join(out_dir, "training_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return model_path

    def train(self) -> Dict[str, Union[float, dict]]:
        """
        Executes full training across configured epochs with early stopping and test evaluation.
        """
        print("=" * 65)
        print(" SafeTV Violence Detection Engine - Training Pipeline")
        print("=" * 65)
        print(f"  Architecture:  {self.config.architecture}")
        print(f"  Device:        {self.device}")
        print(f"  Epochs:        {self.config.epochs} | Batch Size: {self.config.batch_size}")
        print(f"  Learning Rate: {self.config.learning_rate} (Head) | {self.config.backbone_learning_rate} (Backbone)")
        print(f"  Manifest:      {self.config.manifest_csv}")
        print("=" * 65)

        # 1. Load Data
        loaders = create_video_dataloaders(
            manifest_csv_path=self.config.manifest_csv,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]

        print(f"\n[+] Loaded Datasets:")
        print(f"    Train: {len(train_loader.dataset)} clips")
        print(f"    Val:   {len(val_loader.dataset)} clips")
        print(f"    Test:  {len(test_loader.dataset)} clips (unseen holdout)")

        # 2. Build Model & Optimization
        self.build_model()
        self.setup_optimization(train_loader)

        best_val_f1 = -1.0
        best_val_metrics = None
        patience_counter = 0
        history = []

        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            ep_start = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_metrics, _ = self.evaluate(val_loader)
            self.scheduler.step()

            ep_duration = time.time() - ep_start

            print(
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} | Val F1: {val_metrics.f1_score:.4f} | "
                f"Val Acc: {val_metrics.accuracy*100:.1f}% ({ep_duration:.1f}s)"
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_f1": val_metrics.f1_score,
                "val_acc": val_metrics.accuracy,
            })

            # Checkpoint condition: ONLY save when validation F1 strictly improves!
            if val_metrics.f1_score > (best_val_f1 + self.config.min_delta):
                best_val_f1 = val_metrics.f1_score
                best_val_metrics = val_metrics
                patience_counter = 0

                train_metadata = {
                    "total_epochs_trained": epoch,
                    "dataset_counts": {
                        "train_clips": len(train_loader.dataset),
                        "val_clips": len(val_loader.dataset),
                        "test_clips": len(test_loader.dataset),
                    },
                    "training_history": history,
                }
                ckpt_path = self.save_checkpoint(val_metrics, train_metadata)
                print(f"  --> Checkpoint saved (New Best Val F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"\n[!] Early stopping triggered (no improvement in {patience_counter} epochs).")
                    break

        total_training_time = time.time() - start_time
        print(f"\n[+] Training completed in {total_training_time:.1f}s.")

        # 3. Honest Evaluation on UNSEEN Test Set
        print("\n" + "=" * 65)
        print(" Evaluating Best Model on UNSEEN Test Set")
        print("=" * 65)

        # Load best weights
        best_ckpt = os.path.join(self.config.output_dir, "best_model.pth")
        if os.path.exists(best_ckpt):
            self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))

        test_start = time.time()
        test_loss, test_metrics, subcat_perf = self.evaluate(test_loader)
        test_duration = time.time() - test_start
        latency_per_clip = (test_duration / max(1, len(test_loader.dataset))) * 1000.0

        # Model disk size
        model_size_mb = os.path.getsize(best_ckpt) / (1024 * 1024) if os.path.exists(best_ckpt) else 0.0

        print(f"  Accuracy:             {test_metrics.accuracy * 100:.2f}%")
        print(f"  Precision:            {test_metrics.precision * 100:.2f}%")
        print(f"  Recall (Sensitivity): {test_metrics.recall * 100:.2f}%")
        print(f"  Specificity:          {test_metrics.specificity * 100:.2f}%")
        print(f"  Balanced Accuracy:    {test_metrics.balanced_accuracy * 100:.2f}%")
        print(f"  F1-Score:             {test_metrics.f1_score:.4f}")
        print(f"  False Alarm Rate:     {test_metrics.false_alarm_rate * 100:.2f}%")
        print(f"  Inference Latency:    {latency_per_clip:.1f} ms / clip")
        print(f"  Model File Size:      {model_size_mb:.2f} MB")
        print("\nConfusion Matrix:")
        print(f"  True Positives (TP):  {test_metrics.true_positives}")
        print(f"  False Positives (FP): {test_metrics.false_positives}")
        print(f"  True Negatives (TN):  {test_metrics.true_negatives}")
        print(f"  False Negatives (FN): {test_metrics.false_negatives}")

        print("\nPerformance on Subcategories (Difficult Negatives Audit):")
        for subcat, p in subcat_perf.items():
            acc_pct = (p['correct'] / max(1, p['total'])) * 100.0
            print(f"  * {subcat:20s}: {p['correct']}/{p['total']} correct ({acc_pct:.1f}%)")
        print("=" * 65)

        # Update metadata file with test evaluation results
        meta_file = os.path.join(self.config.output_dir, "training_metadata.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                saved_meta = json.load(f)
            saved_meta["unseen_test_evaluation"] = {
                **test_metrics.to_dict(),
                "test_loss": round(test_loss, 4),
                "latency_per_clip_ms": round(latency_per_clip, 2),
                "model_size_mb": round(model_size_mb, 2),
                "subcategory_breakdown": subcat_perf,
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(saved_meta, f, indent=2)

        return {
            "best_val_f1": best_val_f1,
            "test_metrics": test_metrics.to_dict(),
            "checkpoint_path": best_ckpt,
            "subcategory_perf": subcat_perf,
        }
