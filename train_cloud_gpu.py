"""
SafeTV Cloud GPU Training Script (Google Colab / Kaggle / RunPod).
Enables accelerated fine-tuning of R(2+1)D-18 on CUDA hardware with automatic mixed precision.
"""

import argparse
import os
import sys

try:
    import torch
except ImportError:
    torch = None

from safetv.data.downloader import DatasetDownloader
from safetv.trainer import SafeTVTrainer, TrainingConfig


def run_gpu_training():
    parser = argparse.ArgumentParser(description="SafeTV Cloud GPU Training Pipeline")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--manifest", type=str, default="data/annotations/labels.csv", help="Path to labels.csv")
    parser.add_argument("--output-dir", type=str, default="models/violence_detector", help="Output directory")

    args = parser.parse_args()

    if torch is None or not torch.cuda.is_available():
        print("[!] Warning: CUDA GPU not detected. Running in CPU mode.")
        device = "cpu"
    else:
        print(f"[+] CUDA GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        device = "cuda"

    if not os.path.exists(args.manifest):
        print("[*] Generating benchmark dataset for cloud training run...")
        downloader = DatasetDownloader()
        downloader.build_sample_dataset(base_dir="data", num_samples_per_category=6)

    config = TrainingConfig(
        architecture="r2plus1d_18",
        manifest_csv=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size if device == "cuda" else 4,
        learning_rate=args.lr,
        device=device,
        num_threads=6,
    )

    trainer = SafeTVTrainer(config)
    trainer.train()
    print(f"\n[+] GPU Training finished! Checkpoints saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    run_gpu_training()
