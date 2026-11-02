"""
SafeTV Demo: Video Synthesis, Full Inference, and Real-Time HUD Annotation.
Creates a surveillance video file, processes it through the pipeline,
draws a live HUD with probability gauges and status badges, and saves the annotated video.
"""

import json
import os
import cv2
import numpy as np

from safetv.benchmark_generator import SyntheticVideoGenerator
from safetv.pipeline import SafeTVPipeline


def create_demo_video(output_path: str = "demo_surveillance_timeline.mp4", duration_sec: float = 10.0, fps: float = 30.0) -> str:
    """Synthesizes and writes a 10s video with walking, brawl, and dispersal to disk."""
    print(f"[*] Synthesizing {duration_sec}s surveillance scene (Incident between 3.5s - 6.5s)...")
    frames, _ = SyntheticVideoGenerator.generate_timeline_scenario(
        total_sec=duration_sec,
        incident_start_sec=3.5,
        incident_end_sec=6.5,
        fps=fps,
        width=320,
        height=240
    )

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    print(f"[+] Saved input surveillance video to: {os.path.abspath(output_path)}")
    return output_path


def annotate_and_export(
    input_video: str,
    pipeline_report,
    output_video: str = "demo_surveillance_annotated.mp4"
) -> None:
    """Overlays SafeTV detection HUD, probability bars, and incident badges onto the video."""
    print(f"[*] Annotating video with SafeTV Real-Time HUD...")
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # Build timeline lookup by frame index
    timeline = pipeline_report.window_timeline
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        # Find latest active window prediction at or before current time
        matched_window = None
        for w_item in timeline:
            if w_item["start_time_sec"] <= current_time <= w_item["end_time_sec"]:
                matched_window = w_item
                break

        if matched_window:
            prob = matched_window["smoothed_violence_prob"]
            is_incident = matched_window["incident_active"]
        else:
            prob = 0.10
            is_incident = False

        # Draw HUD Header Bar
        cv2.rectangle(frame, (0, 0), (w, 42), (25, 28, 36), -1)
        cv2.putText(frame, "SafeTV Live Surveillance Engine", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(frame, f"Time: {current_time:.2f}s | FPS: {fps:.0f}", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        # Draw Status Badge
        if is_incident or prob >= 0.65:
            badge_color = (40, 40, 220)  # Red (BGR)
            badge_text = "ALERT: VIOLENCE DETECTED"
        elif prob >= 0.45:
            badge_color = (30, 160, 240)  # Amber (BGR)
            badge_text = "MONITORING (TRANSITION)"
        else:
            badge_color = (60, 180, 50)  # Green (BGR)
            badge_text = "STATUS: NORMAL ACTIVITY"

        # Badge rectangle
        cv2.rectangle(frame, (w - 180, 8), (w - 10, 32), badge_color, -1)
        cv2.putText(frame, badge_text, (w - 175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)

        # Draw Real-Time Probability Bar at Bottom
        bar_x, bar_y = 10, h - 22
        bar_w, bar_h = w - 20, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * prob)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), badge_color, -1)
        cv2.putText(frame, f"Violence Confidence: {prob * 100:.1f}%", (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[+] Saved annotated video with HUD to: {os.path.abspath(output_video)}")


def main():
    print("=" * 65)
    print(" SafeTV Full Video Execution & HUD Annotation Demo")
    print("=" * 65)

    input_vid = create_demo_video("demo_surveillance_timeline.mp4")
    pipeline = SafeTVPipeline()

    print("\n[*] Processing video file through SafeTVPipeline...")
    report = pipeline.process_file(input_vid)
    report_json_path = "demo_incident_report.json"
    report.save_json(report_json_path)

    print(f"\n[+] Processing Completed in {report.execution_time_sec:.2f}s!")
    print(f"    Total Video Duration: {report.total_duration_sec:.2f}s")
    print(f"    Windows Analyzed:     {report.total_windows_processed}")
    print(f"    Incidents Detected:   {report.incidents_detected}")
    print(f"    Peak Violence Prob:   {report.peak_violence_prob * 100:.1f}%")

    if report.incidents:
        print("\n--- Detected Incident Details ---")
        for inc in report.incidents:
            print(f"  * Incident #{inc['incident_id']} [{inc['severity']}]:")
            print(f"    Interval: {inc['start_time_sec']}s - {inc['end_time_sec']}s (Duration: {inc['duration_sec']}s)")
            print(f"    Peak Confidence: {inc['peak_probability'] * 100:.1f}% | Mean: {inc['mean_probability'] * 100:.1f}%")

    annotate_and_export(input_vid, report, "demo_surveillance_annotated.mp4")
    print("\n" + "=" * 65)
    print(f" All demo outputs generated successfully!")
    print(f" 1. Raw Surveillance Video:    {os.path.abspath(input_vid)}")
    print(f" 2. Annotated Video with HUD:  {os.path.abspath('demo_surveillance_annotated.mp4')}")
    print(f" 3. Incident JSON Report:      {os.path.abspath(report_json_path)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
