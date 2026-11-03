"""
SafeTV Dataset Preparation CLI.
Automates dataset inspection, validation, partitioning, manifest generation,
and build of benchmark datasets.
Enforces unambiguous directory and label parsing.
"""

import argparse
import json
import os
import sys

from safetv.data.config import DatasetConfig
from safetv.data.downloader import DatasetDownloader
from safetv.data.preprocessor import PreprocessedClip, VideoPreprocessor
from safetv.data.video_inspector import VideoInspector


def parse_video_label(file_path: str) -> tuple[str, int]:
    """
    Explicit, unambiguous directory and path label parsing.
    Prevents substring collision where 'viol' matches inside 'non_violence'.
    Raises ValueError for ambiguous paths.
    """
    p_norm = os.path.normpath(file_path).lower()
    parts = p_norm.split(os.sep)

    # 1. Check explicit folder component names first
    if "non_violence" in parts or "non-violence" in parts or "nonviolence" in parts:
        return "non_violence", 0
    if "violence" in parts:
        return "violence", 1

    # 2. Check filename keywords if directory is not explicitly named
    filename = parts[-1]
    if any(k in filename for k in ["fight", "punch", "kick", "assault", "brawl", "violence"]):
        if not any(neg in filename for neg in ["non", "not", "no_"]):
            return "violence", 1

    if any(k in filename for k in ["walk", "run", "dance", "exercis", "hug", "celebrat", "play", "normal"]):
        return "non_violence", 0

    # 3. Reject ambiguous path
    raise ValueError(f"Ambiguous label for video path '{file_path}'. Cannot determine violence vs non_violence.")


def main():
    parser = argparse.ArgumentParser(description="SafeTV Dataset Preparation and Partitioning Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command 1: info
    subparsers.add_parser("info", help="Display supported academic dataset sources and licensing")

    # Command 2: build-sample
    sample_parser = subparsers.add_parser("build-sample", help="Build a verified sample dataset with violence and non-violence categories")
    sample_parser.add_argument("--output-dir", type=str, default="data", help="Root directory for dataset storage")
    sample_parser.add_argument("--samples-per-cat", type=int, default=4, help="Number of sample videos per category")

    # Command 3: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a raw dataset folder (e.g. RWF-2000 or custom CCTV footage)")
    ingest_parser.add_argument("raw_dir", type=str, help="Path to raw dataset folder containing violence/ and non_violence/ subfolders")
    ingest_parser.add_argument("--dataset-name", type=str, default="custom", help="Identifier name for the dataset")
    ingest_parser.add_argument("--output-dir", type=str, default="data", help="Root directory for dataset storage")

    # Command 4: validate
    val_parser = subparsers.add_parser("validate", help="Validate manifest integrity and verify zero train/test leakage")
    val_parser.add_argument("--manifest", type=str, default="data/annotations/labels.csv", help="Path to labels.csv")

    args = parser.parse_args()

    downloader = DatasetDownloader()

    if args.command == "info" or args.command is None:
        downloader.print_dataset_sources()

    elif args.command == "build-sample":
        print(f"[*] Building benchmark sample dataset in: {args.output_dir}...")
        csv_p, json_p = downloader.build_sample_dataset(
            base_dir=args.output_dir,
            num_samples_per_category=args.samples_per_cat
        )
        print(f"[+] Dataset successfully created!")
        print(f"    Labels CSV: {os.path.abspath(csv_p)}")
        print(f"    Manifest:   {os.path.abspath(json_p)}")

        with open(json_p, "r", encoding="utf-8") as f:
            summary = json.load(f)
        print("\nDataset Summary:")
        print(f"  Total Clips:        {summary['total_clips']}")
        print(f"  Split Distribution: {summary['split_distribution']}")
        print(f"  Label Distribution: {summary['label_distribution']}")
        print(f"  Subcategories:      {list(summary['subcategory_distribution'].keys())}")

    elif args.command == "ingest":
        print(f"[*] Ingesting raw dataset from: {args.raw_dir}...")
        inspector = VideoInspector()
        preprocessor = VideoPreprocessor()

        valid_videos, invalid_videos = inspector.scan_directory(args.raw_dir)
        print(f"[+] Scanned {len(valid_videos) + len(invalid_videos)} videos.")
        print(f"    Valid:   {len(valid_videos)}")
        print(f"    Invalid: {len(invalid_videos)}")

        if invalid_videos:
            print("[!] Warning: Skipped invalid/corrupt files:")
            for inv in invalid_videos[:5]:
                print(f"    - {inv.file_name}: {inv.error_message}")

        entries = []
        skipped_ambiguous = 0
        for v in valid_videos:
            try:
                lbl, lbl_id = parse_video_label(v.file_path)
            except ValueError as e:
                print(f"    [!] Skipping: {e}")
                skipped_ambiguous += 1
                continue

            entries.append({
                "video_path": v.file_path,
                "label": lbl,
                "label_id": lbl_id,
                "subcategory": "surveillance_capture",
                "source_id": os.path.splitext(v.file_name)[0],
                "source_dataset": args.dataset_name,
            })

        if skipped_ambiguous > 0:
            print(f"[!] Skipped {skipped_ambiguous} video(s) due to ambiguous labeling.")

        splits = preprocessor.partition_videos(entries)
        clips = []
        c_idx = 0
        for s_name, s_entries in splits.items():
            for e in s_entries:
                clip = PreprocessedClip(
                    clip_id=f"clip_{c_idx:05d}",
                    source_video_path=e["video_path"],
                    processed_clip_path=e["video_path"],
                    split=s_name,
                    label=e["label"],
                    label_id=e["label_id"],
                    subcategory=e["subcategory"],
                    source_dataset=e["source_dataset"],
                    start_frame=0,
                    end_frame=16,
                    start_sec=0.0,
                    end_sec=2.0,
                    num_frames=16,
                )
                clips.append(clip)
                c_idx += 1

        ann_dir = os.path.join(args.output_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        csv_path = os.path.join(ann_dir, "labels.csv")
        json_path = os.path.join(ann_dir, "manifest.json")

        preprocessor.export_manifest(clips, csv_path, json_path)
        print(f"[+] Ingestion complete: {len(clips)} clips partitioned across train/val/test splits.")
        print(f"    Manifest exported to {csv_path}")

    elif args.command == "validate":
        print(f"[*] Validating manifest at: {args.manifest}...")
        preprocessor = VideoPreprocessor()
        valid, msg = preprocessor.validate_manifest_splits(args.manifest)
        if valid:
            print(f"[+] PASS: {msg}")
        else:
            print(f"[-] FAIL: {msg}")
            sys.exit(1)


if __name__ == "__main__":
    main()
