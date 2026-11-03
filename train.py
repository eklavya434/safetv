"""
SafeTV Training CLI.
Executes spatiotemporal action model fine-tuning, checkpointing, and evaluation.
"""

import argparse
import os
import sys

from safetv.trainer import SafeTVTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="SafeTV Violence Detection Model Training")
    parser.add_argument("--manifest", type=str, default="data/annotations/labels.csv", help="Path to labels.csv")
    parser.add_argument("--output-dir", type=str, default="models/violence_detector", help="Output directory for model weights")
    parser.add_argument("--architecture", type=str, default="r2plus1d_18", choices=["r2plus1d_18", "mc3_18"], help="Backbone architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for classification head")
    parser.add_argument("--backbone-lr", type=float, default=2e-5, help="Learning rate for fine-tuning layer3 & layer4")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Target compute device")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and evaluate existing best_model.pth on test set")

    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[!] Manifest not found at: {args.manifest}")
        print("    Run 'python prepare_dataset.py build-sample' or 'python prepare_dataset.py ingest' first.")
        sys.exit(1)

    config = TrainingConfig(
        architecture=args.architecture,
        manifest_csv=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone_learning_rate=args.backbone_lr,
        device=args.device,
        early_stopping_patience=args.patience,
    )

    trainer = SafeTVTrainer(config)

    if args.evaluate_only:
        ckpt = os.path.join(args.output_dir, "best_model.pth")
        if not os.path.exists(ckpt):
            print(f"[!] Error: Checkpoint not found at: {ckpt}")
            sys.exit(1)
        print(f"[*] Loading model from: {ckpt}")
        trainer.build_model()
        import torch
        trainer.model.load_state_dict(torch.load(ckpt, map_location=trainer.device))

        from safetv.data.dataset import create_video_dataloaders
        loaders = create_video_dataloaders(args.manifest, batch_size=args.batch_size)
        loss, metrics, subcat_perf = trainer.evaluate(loaders["test"])
        print("\n" + "=" * 65)
        print(" Test Set Evaluation Results")
        print("=" * 65)
        print(f"  Accuracy:          {metrics.accuracy * 100:.2f}%")
        print(f"  Precision:         {metrics.precision * 100:.2f}%")
        print(f"  Recall:            {metrics.recall * 100:.2f}%")
        print(f"  Specificity:       {metrics.specificity * 100:.2f}%")
        print(f"  F1-Score:          {metrics.f1_score:.4f}")
        print(f"  False Alarm Rate:  {metrics.false_alarm_rate * 100:.2f}%")
        print("=" * 65)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
