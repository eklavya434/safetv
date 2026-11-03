"""
SafeTV Dataset Preparation CLI.
Automates dataset inspection, validation, partitioning, manifest generation,
and build of benchmark datasets.
"""

import argparse
import json
import os
import sys

from safetv.data.config import DatasetConfig
from safetv.data.downloader import DatasetDownloader
from safetv.data.preprocessor import PreprocessedClip, VideoPreprocessor
from safetv.data.video_inspector import VideoInspector


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
        for v in valid_videos:
            p_lower = v.file_path.lower()
            if "fight" in p_lower or "viol" in p_lower:
                lbl, lbl_id = "violence", 1
            else:
                lbl, lbl_id = "non_violence", 0

            entries.append({
                "video_path": v.file_path,
                "label": lbl,
                "label_id": lbl_id,
                "subcategory": "surveillance_capture",
                "source_id": os.path.splitext(v.file_name)[0],
                "source_dataset": args.dataset_name,
            })

        splits = preprocessor.partition_videos(entries)
        clips = []
        c_idx = 0
        for s_name, s_entries in splits.items():
            for e in s_entries:
                c_idx += 1
                meta = inspector.inspect_file(str(e["video_path"]))
                clips.append(PreprocessedClip(
                    clip_id=f"clip_{c_idx:05d}",
                    source_video_path=str(e["video_path"]),
                    processed_clip_path=str(e["video_path"]),
                    split=s_name,
                    label=str(e["label"]),
                    label_id=int(e["label_id"]),
                    subcategory=str(e["subcategory"]),
                    source_dataset=args.dataset_name,
                    start_frame=0,
                    end_frame=meta.total_frames - 1,
                    start_sec=0.0,
                    end_sec=meta.duration_sec,
                    num_frames=meta.total_frames,
                ))

        csv_p, json_p = preprocessor.write_manifest(
            clips,
            manifest_csv_path=os.path.join(args.output_dir, "annotations", "labels.csv"),
            summary_json_path=os.path.join(args.output_dir, "annotations", "manifest.json")
        )
        print(f"[+] Ingestion complete. Manifest saved to {csv_p}")

    elif args.command == "validate":
        print(f"[*] Validating manifest at: {args.manifest}...")
        if not os.path.exists(args.manifest):
            print(f"[!] Error: Manifest file not found: {args.manifest}")
            sys.exit(1)

        import csv
        split_sources = {"train": set(), "val": set(), "test": set()}
        missing_files = []
        total_rows = 0

        with open(args.manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                v_path = row.get("source_video_path", "")
                split = row.get("split", "")
                if not os.path.exists(v_path):
                    missing_files.append(v_path)
                base = os.path.splitext(os.path.basename(v_path))[0]
                if split in split_sources:
                    split_sources[split].add(base)

        train_val_overlap = split_sources["train"].intersection(split_sources["val"])
        train_test_overlap = split_sources["train"].intersection(split_sources["test"])
        val_test_overlap = split_sources["val"].intersection(split_sources["test"])

        print(f"  Total Clips: {total_rows}")
        print(f"  Missing files on disk: {len(missing_files)}")
        print(f"  Train/Val Leakage:     {len(train_val_overlap)} overlaps")
        print(f"  Train/Test Leakage:    {len(train_test_overlap)} overlaps")
        print(f"  Val/Test Leakage:      {len(val_test_overlap)} overlaps")

        if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0 and len(missing_files) == 0:
            print("\n[+] VALIDATION PASSED: Manifest is clean with 0% data leakage across splits!")
        else:
            print("\n[!] VALIDATION FAILED: Issues found in manifest.")
            sys.exit(1)


if __name__ == "__main__":
    main()
