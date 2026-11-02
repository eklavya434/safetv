"""
SafeTV CLI Entry Point.
Run video violence analysis, benchmarks, and model evaluation from the terminal.
"""

import argparse
import json
import os
import sys

from safetv.benchmark_generator import SyntheticVideoGenerator
from safetv.config import SafeTVConfig
from safetv.metrics import ViolenceDetectionEvaluator
from safetv.pipeline import SafeTVPipeline


def run_benchmark(output_json: str = "benchmark_report.json") -> None:
    """
    Executes reproducible benchmark scenarios evaluating:
    1. Normal walking (benign low energy)
    2. Rhythmic aerobic exercise (benign high energy)
    3. Violent physical altercation (chaotic high impact combat)
    4. Mixed timeline with temporal incident localization
    """
    print("=" * 65)
    print(" SafeTV Video Action Violence Detection Engine - Benchmark Suite")
    print("=" * 65)

    pipeline = SafeTVPipeline()

    print("\n[1/4] Generating & Evaluating Normal Walking Scenario...")
    walking_frames = SyntheticVideoGenerator.generate_normal_walking(num_frames=120, fps=30.0)
    rep_walking = pipeline.process_frames(walking_frames, fps=30.0, source_name="normal_walking")
    print(f"      Duration: {rep_walking.total_duration_sec:.2f}s | Windows: {rep_walking.total_windows_processed}")
    print(f"      Peak Violence Prob: {rep_walking.peak_violence_prob:.4f} | Incidents: {rep_walking.incidents_detected}")

    print("\n[2/4] Generating & Evaluating Rhythmic High-Energy Exercise...")
    exercise_frames = SyntheticVideoGenerator.generate_normal_exercise(num_frames=120, fps=30.0)
    rep_exercise = pipeline.process_frames(exercise_frames, fps=30.0, source_name="normal_exercise")
    print(f"      Duration: {rep_exercise.total_duration_sec:.2f}s | Windows: {rep_exercise.total_windows_processed}")
    print(f"      Peak Violence Prob: {rep_exercise.peak_violence_prob:.4f} | Incidents: {rep_exercise.incidents_detected}")

    print("\n[3/4] Generating & Evaluating Violent Physical Brawl...")
    brawl_frames = SyntheticVideoGenerator.generate_violent_brawl(num_frames=120, fps=30.0)
    rep_brawl = pipeline.process_frames(brawl_frames, fps=30.0, source_name="violent_brawl")
    print(f"      Duration: {rep_brawl.total_duration_sec:.2f}s | Windows: {rep_brawl.total_windows_processed}")
    print(f"      Peak Violence Prob: {rep_brawl.peak_violence_prob:.4f} | Incidents: {rep_brawl.incidents_detected}")

    print("\n[4/4] Generating & Evaluating 10s Timeline Incident Localization...")
    gt_start = 3.5
    gt_end = 6.5
    timeline_frames, gt_interval = SyntheticVideoGenerator.generate_timeline_scenario(
        total_sec=10.0,
        incident_start_sec=gt_start,
        incident_end_sec=gt_end,
        fps=30.0
    )
    rep_timeline = pipeline.process_frames(timeline_frames, fps=30.0, source_name="mixed_timeline")
    print(f"      Duration: {rep_timeline.total_duration_sec:.2f}s | Windows: {rep_timeline.total_windows_processed}")
    print(f"      Peak Violence Prob: {rep_timeline.peak_violence_prob:.4f} | Incidents: {rep_timeline.incidents_detected}")

    # Compute window-level ground truth and predictions for the timeline scenario
    y_true = []
    y_pred = []
    y_probs = []
    for w in rep_timeline.window_timeline:
        center = w["center_time_sec"]
        is_violence_gt = 1 if (gt_start <= center <= gt_end) else 0
        y_true.append(is_violence_gt)
        y_pred.append(1 if w["smoothed_violence_prob"] >= pipeline.config.incident.onset_threshold else 0)
        y_probs.append(w["smoothed_violence_prob"])

    eval_report = ViolenceDetectionEvaluator.compute_window_metrics(y_true, y_pred, y_probs)

    # Compute temporal IoU
    tiou = 0.0
    if rep_timeline.incidents:
        first_inc = rep_timeline.incidents[0]
        tiou = ViolenceDetectionEvaluator.compute_temporal_iou(
            (first_inc["start_time_sec"], first_inc["end_time_sec"]),
            gt_interval
        )
    eval_report.mean_temporal_iou = tiou

    print("\n" + "=" * 65)
    print(" SafeTV Quantitative Performance Metrics")
    print("=" * 65)
    print(f"  Accuracy:             {eval_report.accuracy * 100:.2f}%")
    print(f"  Precision:            {eval_report.precision * 100:.2f}%")
    print(f"  Recall (Sensitivity): {eval_report.recall * 100:.2f}%")
    print(f"  Specificity:          {eval_report.specificity * 100:.2f}%")
    print(f"  Balanced Accuracy:    {eval_report.balanced_accuracy * 100:.2f}%")
    print(f"  F1-Score:             {eval_report.f1_score:.4f}")
    print(f"  False Alarm Rate:     {eval_report.false_alarm_rate * 100:.2f}%")
    print(f"  Temporal IoU (tIoU):  {tiou:.4f}")
    print("=" * 65)

    benchmark_summary = {
        "metrics": eval_report.to_dict(),
        "scenarios": {
            "normal_walking": rep_walking.to_dict(),
            "normal_exercise": rep_exercise.to_dict(),
            "violent_brawl": rep_brawl.to_dict(),
            "mixed_timeline": rep_timeline.to_dict(),
        }
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"\n[+] Full benchmark results saved to: {os.path.abspath(output_json)}")


def main():
    parser = argparse.ArgumentParser(description="SafeTV Video Violence Detection Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    bench_parser = subparsers.add_parser("benchmark", help="Run reproducible synthetic benchmark suite")
    bench_parser.add_argument("--output", type=str, default="benchmark_report.json", help="Path to save output JSON")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a video file")
    analyze_parser.add_argument("video_path", type=str, help="Path to input video file")
    analyze_parser.add_argument("--output", type=str, default="incident_report.json", help="Path to save output JSON")
    analyze_parser.add_argument("--max-duration", type=float, default=None, help="Max duration in seconds to analyze")

    args = parser.parse_args()

    if args.command == "benchmark" or args.command is None:
        out = getattr(args, "output", "benchmark_report.json")
        run_benchmark(output_json=out)
    elif args.command == "analyze":
        pipeline = SafeTVPipeline()
        print(f"[*] Analyzing video: {args.video_path}")
        report = pipeline.process_file(args.video_path, max_duration_sec=args.max_duration)
        report.save_json(args.output)
        print(f"[+] Processed {report.total_windows_processed} windows in {report.execution_time_sec:.2f}s")
        print(f"[+] Detected {report.incidents_detected} incident(s). Report saved to {args.output}")


if __name__ == "__main__":
    main()
