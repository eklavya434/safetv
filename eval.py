"""
SafeTV Model Evaluation Script.
Computes comprehensive, honest evaluation metrics on held-out test splits:
- ROC-AUC and PR-AUC
- Expected Calibration Error (ECE) and Brier Score
- Confusion Matrix (TP, FP, TN, FN)
- False Alarm Rate at fixed True Positive Rate (FAR @ TPR=0.90, FAR @ TPR=0.95)
- Latency and throughput benchmarks
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
except ImportError as e:
    print(f"[!] Error importing dependencies: {e}")
    sys.exit(1)

from safetv.calibration import ProbabilityCalibrator
from safetv.data.dataset import create_video_dataloaders
from safetv.metrics import ViolenceDetectionEvaluator
from safetv.model import VideoActionClassifier
from safetv.trainer import SafeTVTrainer, TrainingConfig


def compute_far_at_tpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_tprs: Tuple[float, ...] = (0.90, 0.95)
) -> Dict[str, float]:
    """
    Computes False Alarm Rate (FPR) at fixed target True Positive Rates.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    results = {}
    for target in target_tprs:
        # Find smallest threshold index where TPR >= target
        valid_indices = np.where(tpr >= target)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[0]
            far = float(fpr[idx])
            thresh = float(thresholds[idx])
            results[f"far_at_tpr_{int(target * 100)}"] = round(far, 4)
            results[f"threshold_at_tpr_{int(target * 100)}"] = round(thresh, 4)
        else:
            results[f"far_at_tpr_{int(target * 100)}"] = 1.0
            results[f"threshold_at_tpr_{int(target * 100)}"] = 0.0
    return results


def evaluate_test_set(
    manifest_path: str,
    checkpoint_path: str,
    batch_size: int = 4,
    device_str: str = "cpu",
    output_json: Optional[str] = None
) -> Dict:
    """
    Runs full test set evaluation and returns quantitative metrics dictionary.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    print("=" * 70)
    print(" SafeTV Model Evaluation Suite")
    print(f" Manifest:   {manifest_path}")
    print(f" Checkpoint: {checkpoint_path}")
    print(f" Device:     {device}")
    print("=" * 70)

    # 1. Load Data
    loaders = create_video_dataloaders(
        manifest_csv_path=manifest_path,
        batch_size=batch_size,
        num_workers=0
    )
    test_loader = loaders.get("test")
    if test_loader is None or len(test_loader.dataset) == 0:
        raise ValueError("No test clips found in manifest!")

    print(f"[+] Loaded test split with {len(test_loader.dataset)} clips.")

    # 2. Build Model & Load Checkpoint
    config = TrainingConfig(
        architecture="r2plus1d_18",
        manifest_csv=manifest_path,
        output_dir=os.path.dirname(checkpoint_path),
        device=device_str,
    )
    trainer = SafeTVTrainer(config)
    trainer.build_model()

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    trainer.model.load_state_dict(state, strict=True)
    trainer.model.eval()

    # Fit temperature scaling on validation split
    cal_file = os.path.join(os.path.dirname(checkpoint_path), "calibration.json")
    val_loader = loaders.get("val")
    calibrator = ProbabilityCalibrator(calibration_file=cal_file)
    if val_loader and len(val_loader.dataset) > 0:
        val_logits, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                v = batch["video"].to(device)
                l = batch["label"].to(device)
                val_logits.extend(trainer.model(v).cpu().numpy())
                val_labels.extend(l.cpu().numpy())
        opt_t = calibrator.fit(np.array(val_logits), np.array(val_labels))
        print(f"[+] Fitted temperature scaling on validation split: T = {opt_t:.4f}")

        # Store in metadata
        meta_path = os.path.join(os.path.dirname(checkpoint_path), "training_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["calibration"] = {
                    "temperature": opt_t,
                    "validation_clips_used": len(val_labels),
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                print(f"[-] Warning updating training_metadata.json: {e}")

    # 3. Inference Loop
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []
    all_logits: List[List[float]] = []

    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            videos = batch["video"].to(device)
            labels = batch["label"].to(device)

            logits = trainer.model(videos)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_logits.extend(logits.cpu().tolist())

    elapsed_time = time.time() - start_time
    total_clips = len(all_labels)
    latency_ms = (elapsed_time / max(1, total_clips)) * 1000.0

    y_true = np.array(all_labels, dtype=np.int32)
    y_pred = np.array(all_preds, dtype=np.int32)
    y_prob = np.array(all_probs, dtype=np.float32)

    # 4. Standard Classification Metrics
    base_metrics = ViolenceDetectionEvaluator.compute_window_metrics(all_labels, all_preds, all_probs)

    # 5. Calibration & Discrimination
    cal_metrics = ProbabilityCalibrator.compute_ece(all_labels, all_probs, num_bins=10)
    brier_score = ProbabilityCalibrator.compute_brier_score(all_labels, all_probs)

    calibrated_probs_arr = calibrator.calibrate_logits(np.array(all_logits))[:, 1]
    cal_metrics_scaled = ProbabilityCalibrator.compute_ece(all_labels, calibrated_probs_arr.tolist(), num_bins=10)
    brier_score_scaled = ProbabilityCalibrator.compute_brier_score(all_labels, calibrated_probs_arr.tolist())

    # ROC-AUC & PR-AUC
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        far_metrics = compute_far_at_tpr(y_true, y_prob, target_tprs=(0.90, 0.95))
    else:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))
        far_metrics = {"far_at_tpr_90": 0.0, "far_at_tpr_95": 0.0}

    results = {
        "manifest": manifest_path,
        "checkpoint": checkpoint_path,
        "total_test_samples": total_clips,
        "confusion_matrix": {
            "true_positives": base_metrics.true_positives,
            "false_positives": base_metrics.false_positives,
            "true_negatives": base_metrics.true_negatives,
            "false_negatives": base_metrics.false_negatives,
        },
        "accuracy": round(base_metrics.accuracy, 4),
        "precision": round(base_metrics.precision, 4),
        "recall_sensitivity": round(base_metrics.recall, 4),
        "specificity": round(base_metrics.specificity, 4),
        "balanced_accuracy": round(base_metrics.balanced_accuracy, 4),
        "f1_score": round(base_metrics.f1_score, 4),
        "false_alarm_rate": round(base_metrics.false_alarm_rate, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "temperature": round(calibrator.temperature, 4),
        "raw_brier_score": round(brier_score, 4),
        "raw_ece": round(cal_metrics.expected_calibration_error, 4),
        "calibrated_brier_score": round(brier_score_scaled, 4),
        "calibrated_ece": round(cal_metrics_scaled.expected_calibration_error, 4),
        **far_metrics,
        "latency_per_clip_ms": round(latency_ms, 2),
        "total_evaluation_sec": round(elapsed_time, 2),
    }

    # 6. Formatted Output
    print("\n" + "=" * 70)
    print(" HELD-OUT TEST SPLIT EVALUATION RESULTS")
    print("=" * 70)
    print(f" Total Clips Evaluated:   {total_clips}")
    print(f" Accuracy:               {results['accuracy'] * 100:.2f}%")
    print(f" Precision:              {results['precision'] * 100:.2f}%")
    print(f" Recall / Sensitivity:   {results['recall_sensitivity'] * 100:.2f}%")
    print(f" Specificity:            {results['specificity'] * 100:.2f}%")
    print(f" Balanced Accuracy:      {results['balanced_accuracy'] * 100:.2f}%")
    print(f" F1-Score:               {results['f1_score']:.4f}")
    print(f" False Alarm Rate:       {results['false_alarm_rate'] * 100:.2f}%")
    print("-" * 70)
    print(f" ROC-AUC:                {results['roc_auc']:.4f}")
    print(f" PR-AUC:                 {results['pr_auc']:.4f}")
    print(f" Temperature (T):        {results['temperature']:.4f}")
    print(f" Raw Brier Score:        {results['raw_brier_score']:.4f}  (lower is better)")
    print(f" Raw ECE:                {results['raw_ece']:.4f}  (Expected Calibration Error)")
    print(f" Calibrated Brier Score: {results['calibrated_brier_score']:.4f}  (lower is better)")
    print(f" Calibrated ECE:         {results['calibrated_ece']:.4f}  (lower is better)")
    if "far_at_tpr_90" in results:
        print(f" FAR @ TPR = 90%:        {results['far_at_tpr_90'] * 100:.2f}%")
    if "far_at_tpr_95" in results:
        print(f" FAR @ TPR = 95%:        {results['far_at_tpr_95'] * 100:.2f}%")
    print("-" * 70)
    print(" Confusion Matrix:")
    print(f"   True Positives (TP):  {results['confusion_matrix']['true_positives']}")
    print(f"   False Positives (FP): {results['confusion_matrix']['false_positives']}")
    print(f"   True Negatives (TN):  {results['confusion_matrix']['true_negatives']}")
    print(f"   False Negatives (FN): {results['confusion_matrix']['false_negatives']}")
    print(f" Mean Latency:           {results['latency_per_clip_ms']:.1f} ms / clip")
    print("=" * 70)

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[+] Saved evaluation metrics to: {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SafeTV Honest Model Evaluation CLI")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest labels.csv")
    parser.add_argument("--checkpoint", type=str, default="models/violence_detector/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path")

    args = parser.parse_args()

    # Auto-resolve manifest
    manifest = args.manifest
    if not manifest:
        if os.path.exists("data/real_dataset/annotations/labels.csv"):
            manifest = "data/real_dataset/annotations/labels.csv"
        elif os.path.exists("data/annotations/labels.csv"):
            manifest = "data/annotations/labels.csv"
        else:
            print("[!] Error: No labels.csv manifest found.")
            sys.exit(1)

    evaluate_test_set(
        manifest_path=manifest,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device_str=args.device,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
