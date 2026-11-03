"""
Diagnostic harness to evaluate impact of input preprocessing variants on
violence classification probability scores.
Systematically tests:
1. Color space: RGB vs BGR
2. Pixel scaling: [0.0, 1.0] float vs [0.0, 255.0] unscaled
3. Normalization: Kinetics-400 vs ImageNet vs Zero-mean vs None
4. Sampling temporal stride: 2 vs 4 vs 8 vs 16
5. Clip length: 8 vs 16 vs 32 frames
6. Spatial aspect ratio: Bilinear full-frame resize vs Aspect-preserving center-crop
7. Model execution mode: eval() vs train() (dropout active)
8. Output head mapping: Logit index 1 (violence) vs Logit index 0 (non_violence / swapped)
"""

import argparse
import os
import sys
import numpy as np
import torch
import cv2

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetv.config import SafeTVConfig
from safetv.model import VideoActionClassifier


def extract_raw_frames(video_path: str, max_frames: int = 120) -> list[np.ndarray]:
    """Read uncompressed BGR frames from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames could be extracted from: {video_path}")
    return frames


def build_tensor(
    frames: list[np.ndarray],
    num_frames: int = 16,
    stride: int = 8,
    color_order: str = "RGB",
    scale_01: bool = True,
    norm_type: str = "kinetics",
    resize_mode: str = "full_resize",
    target_hw: tuple[int, int] = (112, 112),
) -> torch.Tensor:
    """Builds (1, 3, T, H, W) tensor under specified preprocessing variant."""
    h_target, w_target = target_hw
    # Sample indices
    indices = [min(len(frames) - 1, i * stride) for i in range(num_frames)]
    sampled_bgr = [frames[i] for i in indices]

    processed = []
    for bgr in sampled_bgr:
        # 1. Color space
        if color_order == "RGB":
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            img = bgr.copy()

        # 2. Spatial resize
        if resize_mode == "center_crop":
            # Scale shorter side to 128, then center crop to 112x112
            orig_h, orig_w = img.shape[:2]
            scale = 128.0 / min(orig_h, orig_w)
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            y_start = max(0, (new_h - h_target) // 2)
            x_start = max(0, (new_w - w_target) // 2)
            img = resized[y_start : y_start + h_target, x_start : x_start + w_target]
        else:
            img = cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_LINEAR)

        # 3. Scaling
        arr = img.astype(np.float32)
        if scale_01:
            arr = arr / 255.0

        # (H, W, C) -> (C, H, W)
        arr = np.transpose(arr, (2, 0, 1))

        # 4. Normalization
        if norm_type == "kinetics":
            mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32).reshape(3, 1, 1)
            arr = (arr - mean) / std
        elif norm_type == "imagenet":
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            arr = (arr - mean) / std
        elif norm_type == "zero_mean":
            arr = (arr - 0.5) / 0.5
        elif norm_type == "none":
            pass

        processed.append(arr)

    # Stack to (T, C, H, W) -> transpose to (C, T, H, W) -> unsqueeze to (1, C, T, H, W)
    t = np.stack(processed, axis=0)
    t = np.transpose(t, (1, 0, 2, 3))
    tensor = torch.from_numpy(t).unsqueeze(0).float()
    return tensor


def run_diagnostics(video_path: str):
    print("=" * 80)
    print(f" SafeTV Preprocessing & Model Sensitivity Diagnostic")
    print(f" Target Video: {video_path}")
    print("=" * 80)

    config = SafeTVConfig()
    classifier = VideoActionClassifier(config.model)
    print(f"Model Architecture: {classifier.architecture}")
    print(f"Loaded Checkpoint:  {classifier.loaded_checkpoint}")
    print(f"Checkpoint SHA-256: {classifier.checkpoint_hash}")
    print(f"Class Mapping:      {classifier.class_mapping}")
    print(f"Model Is Ready:     {classifier.is_ready}")
    print("-" * 80)

    raw_frames = extract_raw_frames(video_path, max_frames=150)
    print(f"Extracted {len(raw_frames)} raw test frames.")

    experiments = [
        # (Name, Description, kwargs, eval_mode, class_idx)
        (
            "BASELINE (Config Standard)",
            "RGB, [0,1], Kinetics norm, stride=8, T=16, center_crop, eval()",
            dict(num_frames=16, stride=8, color_order="RGB", scale_01=True, norm_type="kinetics", resize_mode="center_crop"),
            True,
            1,
        ),
        (
            "FULL-FRAME RESIZE",
            "RGB, [0,1], Kinetics norm, stride=8, T=16, direct 112x112 resize",
            dict(num_frames=16, stride=8, color_order="RGB", scale_01=True, norm_type="kinetics", resize_mode="full_resize"),
            True,
            1,
        ),
        (
            "COLOR BUG: BGR Order",
            "BGR instead of RGB (unconverted cv2 frames)",
            dict(num_frames=16, stride=8, color_order="BGR", scale_01=True, norm_type="kinetics", resize_mode="center_crop"),
            True,
            1,
        ),
        (
            "SCALING BUG: No [0,1] Div",
            "[0, 255] raw pixel intensities fed to Kinetics norm",
            dict(num_frames=16, stride=8, color_order="RGB", scale_01=False, norm_type="kinetics", resize_mode="center_crop"),
            True,
            1,
        ),
        (
            "NORM: ImageNet Constants",
            "ImageNet mean/std instead of Kinetics video constants",
            dict(num_frames=16, stride=8, color_order="RGB", scale_01=True, norm_type="imagenet", resize_mode="center_crop"),
            True,
            1,
        ),
        (
            "NORM: None (Unnormalized)",
            "Only [0, 1] scaling without mean subtraction or std division",
            dict(num_frames=16, stride=8, color_order="RGB", scale_01=True, norm_type="none", resize_mode="center_crop"),
            True,
            1,
        ),
        (
            "STRIDE: Rapid (stride=2)",
            "High temporal overlap / tight frame delta",
            dict(num_frames=16, stride=2, color_order="RGB", scale_01=True, norm_type="kinetics", resize_mode="center_crop"),
            True,
            1,
        ),
        (
            "STRIDE: Sparse (stride=16)",
            "Wide frame delta spanning 240 frames (~8 sec)",
            dict(num_frames=16, stride=16, color_order="RGB", scale_01=True, norm_type="kinetics", resize_mode="center_crop"),
            True,
            1,
        ),
        (
            "CLIP LENGTH: T=8 frames",
            "Truncated window (interpolated to model expectation)",
            dict(num_frames=8, stride=4, color_order="RGB", scale_01=True, norm_type="kinetics", resize_mode="center_crop"),
            True,
            1,
        ),
        (
            "MODE: Train mode (Dropout ON)",
            "eval() disabled: mc_dropout active during inference",
            dict(num_frames=16, stride=8, color_order="RGB", scale_01=True, norm_type="kinetics", resize_mode="center_crop"),
            False,
            1,
        ),
        (
            "SWAPPED HEAD INDEX",
            "Report class 0 (non_violence logit) instead of class 1",
            dict(num_frames=16, stride=8, color_order="RGB", scale_01=True, norm_type="kinetics", resize_mode="center_crop"),
            True,
            0,
        ),
    ]

    results = []
    print(f"\n{'Variant Name':<28} | {'Mode':<6} | {'Head Idx':<8} | {'Raw Logits (0, 1)':<22} | {'Violence Prob':<14} | Impact")
    print("-" * 105)

    base_prob = None

    for name, desc, kwargs, is_eval, class_idx in experiments:
        tensor = build_tensor(raw_frames, **kwargs)

        # Handle T=8 dimension mismatch by repeat padding to 16
        if tensor.shape[2] != 16:
            tensor = torch.nn.functional.interpolate(tensor, size=(16, 112, 112), mode="trilinear", align_corners=False)

        if is_eval:
            classifier.model.eval()
        else:
            classifier.model.train()

        with torch.no_grad():
            logits = classifier.model(tensor.to(classifier.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        prob = float(probs[class_idx])
        l0 = float(logits[0, 0].item())
        l1 = float(logits[0, 1].item())
        logits_str = f"[{l0:+.2f}, {l1:+.2f}]"

        if base_prob is None:
            base_prob = prob
            delta_str = "BASELINE"
        else:
            delta = prob - base_prob
            delta_str = f"{delta:+.1%}"

        print(f"{name:<28} | {('eval' if is_eval else 'train'):<6} | {class_idx:<8} | {logits_str:<22} | {prob*100:>6.2f}%        | {delta_str}")
        results.append({
            "name": name,
            "description": desc,
            "eval_mode": is_eval,
            "head_idx": class_idx,
            "logit_0": l0,
            "logit_1": l1,
            "prob": prob,
            "delta_from_baseline": prob - (base_prob or 0.0),
        })

    print("-" * 105)
    print("\n[+] SUMMARY OF ROOT CAUSE FINDINGS:")
    swapped = [r for r in results if r["name"] == "SWAPPED HEAD INDEX"][0]
    bgr = [r for r in results if r["name"] == "COLOR BUG: BGR Order"][0]
    unscaled = [r for r in results if r["name"] == "SCALING BUG: No [0,1] Div"][0]
    no_norm = [r for r in results if r["name"] == "NORM: None (Unnormalized)"][0]

    print(f"1. Swapped Head Mapping: Class 0 gives {swapped['prob']*100:.2f}% while Class 1 gives {base_prob*100:.2f}%.")
    print(f"2. BGR Color Inversion: Drops probability by {bgr['delta_from_baseline']:+.1%}.")
    print(f"3. Unscaled [0, 255] Pixels: Severe distribution shift -> score drops to {unscaled['prob']*100:.2f}%.")
    print(f"4. Missing Normalization: Score drops to {no_norm['prob']*100:.2f}%.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose video preprocessing variants on model scores.")
    parser.add_argument(
        "--video",
        type=str,
        default="data/real_dataset/violence/real_fight_09.mp4",
        help="Path to real fight test video."
    )
    args = parser.parse_args()
    run_diagnostics(args.video)
