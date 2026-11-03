"""
SafeTV Diagnostic Evaluation Suite.
Performs an empirical evaluation across 8 test scenarios:
1. Actual physical fight
2. Person walking
3. Two people hugging
4. People exercising
5. People dancing
6. People arguing without physical contact
7. Person running
8. Non-violent crowd interaction

Exports detailed per-window metrics, mean/max probabilities,
predicted labels, and model provenance to CSV.
"""

import csv
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from safetv.pipeline import SafeTVPipeline
from safetv.model import VideoActionClassifier


def run_diagnostic_evaluation(output_csv: str = "diagnostic_report.csv") -> List[Dict]:
    """
    Executes diagnostic evaluation on test scenarios and writes structured CSV report.
    """
    pipeline = SafeTVPipeline()

    scenarios = [
        {
            "category": "actual_physical_fight",
            "ground_truth": "VIOLENCE",
            "path": "uploads/1793688630_fight_0003.mp4",
            "description": "Real-world outdoor night physical fight / grappling altercation"
        },
        {
            "category": "corridor_walking",
            "ground_truth": "NON_VIOLENCE",
            "path": "uploads/1793684301_gemini_generated_video_d74e41c5.mp4",
            "description": "Real-world students walking in school hallway"
        },
        {
            "category": "two_people_hugging",
            "ground_truth": "NON_VIOLENCE",
            "path": "data/raw/non_violence/hugging_001.mp4",
            "description": "Two individuals embracing / hugging"
        },
        {
            "category": "people_exercising",
            "ground_truth": "NON_VIOLENCE",
            "path": "data/raw/non_violence/exercising_001.mp4",
            "description": "High-cadence aerobic / calisthenics exercise"
        },
        {
            "category": "people_dancing",
            "ground_truth": "NON_VIOLENCE",
            "path": "data/raw/non_violence/dancing_001.mp4",
            "description": "Partnered rhythmic dance movements"
        },
        {
            "category": "people_arguing_no_contact",
            "ground_truth": "NON_VIOLENCE",
            "path": "data/raw/non_violence/celebration_001.mp4",
            "description": "Verbal confrontation / emotional gesture without physical strikes"
        },
        {
            "category": "person_running",
            "ground_truth": "NON_VIOLENCE",
            "path": "data/raw/non_violence/running_001.mp4",
            "description": "Rapid locomotion / jogging / running"
        },
        {
            "category": "crowd_interaction",
            "ground_truth": "NON_VIOLENCE",
            "path": "data/raw/non_violence/normal_walking_001.mp4",
            "description": "Multiple individuals navigating shared pedestrian space"
        },
    ]

    results = []

    print("=" * 80)
    print(" SafeTV Diagnostic Evaluation Running")
    print(f" Model Provenance: {pipeline.model.provenance} | Checkpoint: {pipeline.model.loaded_checkpoint}")
    print("=" * 80)

    for sc in scenarios:
        path = sc["path"]
        cat = sc["category"]
        gt = sc["ground_truth"]

        if not os.path.exists(path):
            print(f"[-] Warning: Test video not found: {path} (skipping)", flush=True)
            continue

        print(f"[*] Processing {cat}: {os.path.basename(path)}...", flush=True)
        start_time = time.time()
        # Cap max_duration_sec for very long benchmark videos to 20s for deterministic evaluation
        report = pipeline.process_file(path, max_duration_sec=20.0)
        elapsed = time.time() - start_time

        timeline = report.window_timeline
        raw_probs = [w["raw_violence_prob"] for w in timeline]
        smoothed_probs = [w["smoothed_violence_prob"] for w in timeline]

        mean_prob = report.mean_violence_prob
        peak_prob = report.peak_violence_prob
        incidents_detected = len(report.incidents)
        predicted_class = "VIOLENCE" if peak_prob >= pipeline.config.incident.onset_threshold else "NON_VIOLENCE"

        record = {
            "category": cat,
            "filename": os.path.basename(path),
            "ground_truth": gt,
            "description": sc["description"],
            "total_frames": report.total_frames if hasattr(report, "total_frames") else len(timeline)*8,
            "fps": report.fps,
            "num_windows": len(timeline),
            "mean_violence_prob": round(mean_prob, 4),
            "max_violence_prob": round(peak_prob, 4),
            "predicted_class": predicted_class,
            "incident_detected": incidents_detected > 0,
            "incidents_count": incidents_detected,
            "is_correct": (predicted_class == gt),
            "model_provenance": pipeline.model.provenance,
            "checkpoint_hash": pipeline.model.checkpoint_hash,
            "processing_time_sec": round(elapsed, 2),
            "all_window_probs": ";".join([f"{p:.3f}" for p in raw_probs])
        }
        results.append(record)

        print(f"[+] [{cat}] {os.path.basename(path)}", flush=True)
        print(f"    GT: {gt} | Pred: {predicted_class} | Peak: {peak_prob * 100:.1f}% | Correct: {record['is_correct']}", flush=True)
        print(f"    Windows: {len(timeline)} | Elapsed: {elapsed:.2f}s", flush=True)
        print("-" * 60, flush=True)

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = [
        "category", "filename", "ground_truth", "description",
        "total_frames", "fps", "num_windows",
        "mean_violence_prob", "max_violence_prob", "predicted_class",
        "incident_detected", "incidents_count", "is_correct",
        "model_provenance", "checkpoint_hash", "processing_time_sec",
        "all_window_probs"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"[+] Diagnostic evaluation complete. Report saved to: {output_csv}")
    return results


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "diagnostic_report.csv"
    run_diagnostic_evaluation(out)
