"""
SafeTV Environmental Robustness CLI.
Runs degradation stress-testing across poor lighting, low resolution,
motion blur, and sensor noise using genuine surveillance test videos.
"""

import argparse
import json
import os
import cv2

from safetv.pipeline import SafeTVPipeline
from safetv.robustness import RobustnessEvaluator


def load_video_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    return frames, fps


def main():
    parser = argparse.ArgumentParser(description="SafeTV Robustness & Environmental Stress Tester")
    parser.add_argument("--violence-video", type=str, default="data/raw/violence/punching_001.mp4")
    parser.add_argument("--normal-video", type=str, default="data/raw/non_violence/normal_walking_001.mp4")
    parser.add_argument("--output", type=str, default="robustness_report.json")
    args = parser.parse_args()

    print("=" * 65)
    print(" SafeTV Environmental Robustness & Stress-Testing Suite")
    print("=" * 65)
    print(f"  Violence Video: {args.violence_video}")
    print(f"  Normal Video:   {args.normal_video}")
    print("=" * 65)

    if not os.path.exists(args.violence_video) or not os.path.exists(args.normal_video):
        print("[!] Input video files not found. Please ensure test videos exist.")
        return

    viol_frames, fps_v = load_video_frames(args.violence_video)
    norm_frames, fps_n = load_video_frames(args.normal_video)

    evaluator = RobustnessEvaluator(SafeTVPipeline())
    results = evaluator.run_full_stress_test(viol_frames, norm_frames, fps=fps_v)

    print("\n--- [1] VIOLENCE DETECTION UNDER DEGRADATION ---")
    for name, data in results["violence_detection_under_stress"].items():
        detected = "YES" if data["incidents_detected"] > 0 else "MISSED"
        print(f"  * {name:24s} | Peak Prob: {data['peak_probability']:.4f} | Detected: {detected}")

    print("\n--- [2] FALSE ALARM REJECTION UNDER DEGRADATION ---")
    for name, data in results["normal_rejection_under_stress"].items():
        status = "PASSED (No Alarm)" if data["incidents_detected"] == 0 else "FALSE ALARM"
        print(f"  * {name:24s} | Peak Prob: {data['peak_probability']:.4f} | Result: {status}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Full robustness report saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
