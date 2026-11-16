"""Reproducible 4-class semantic-segmentation evaluation.

The primary protocol is fixed: input logits are converted by argmax over the
four channels [background, glomerulus, vessel, tubule], and labels are scored
in one explicitly selected pixel space.  This script never thresholds or
post-processes a prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


CLASS_IDS = (1, 2, 3)
CLASS_NAMES = {0: "background", 1: "glomerulus", 2: "vessel", 3: "tubule"}


def resize_label(mask: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """Resize a discrete class map with nearest-neighbor interpolation only."""
    if size is None or mask.shape[:2] == size:
        return mask.astype(np.uint8, copy=False)
    return cv2.resize(mask.astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


def binary_boundary(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """Return a width-pixel inner/outer morphological boundary."""
    binary = mask.astype(np.uint8)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(binary, kernel, iterations=width)
    dilated = cv2.dilate(binary, kernel, iterations=width)
    return (dilated != eroded).astype(bool)


def dice_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, int, int, int]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    if tp == fp == fn == 0:
        return 1.0, 1.0, tp, fp, fn
    dice = 2.0 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return dice, iou, tp, fp, fn


def boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance: int) -> float:
    """Symmetric boundary F1 with a tolerance measured in pixels."""
    pred_b = binary_boundary(pred)
    target_b = binary_boundary(target)
    if not pred_b.any() and not target_b.any():
        return 1.0
    if not pred_b.any() or not target_b.any():
        return 0.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    pred_match = np.logical_and(pred_b, cv2.dilate(target_b.astype(np.uint8), kernel).astype(bool)).sum()
    target_match = np.logical_and(target_b, cv2.dilate(pred_b.astype(np.uint8), kernel).astype(bool)).sum()
    precision = pred_match / max(pred_b.sum(), 1)
    recall = target_match / max(target_b.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-12))


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """Symmetric HD95 on binary boundaries; NaN is used when only one is empty."""
    pred_b = binary_boundary(pred)
    target_b = binary_boundary(target)
    if not pred_b.any() and not target_b.any():
        return 0.0
    if not pred_b.any() or not target_b.any():
        return float("nan")
    # distanceTransform computes distance to zero pixels.  Invert the target
    # boundary so target-boundary pixels become zeros.
    d_to_target = cv2.distanceTransform((~target_b).astype(np.uint8), cv2.DIST_L2, 5)
    d_to_pred = cv2.distanceTransform((~pred_b).astype(np.uint8), cv2.DIST_L2, 5)
    distances = np.concatenate((d_to_target[pred_b], d_to_pred[target_b]))
    return float(np.percentile(distances, 95))


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = 4) -> np.ndarray:
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    encoded = num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def evaluate_pair(pred: np.ndarray, target: np.ndarray, filename: str = "") -> dict[str, float | str]:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/GT shape mismatch for {filename}: {pred.shape} vs {target.shape}")
    row: dict[str, float | str] = {"filename": filename}
    matrix = confusion_matrix(pred, target)
    # For single-label multiclass masks, micro TP/FP/FN aggregate to one
    # confusion count and both micro Dice and micro IoU equal pixel accuracy.
    pixel_accuracy = float(np.trace(matrix) / max(matrix.sum(), 1))
    row["micro_dice_including_background"] = pixel_accuracy
    row["micro_iou_including_background"] = pixel_accuracy
    row["pixel_accuracy"] = pixel_accuracy
    dices, ious, bf1_1, bf1_3, hd95s = [], [], [], [], []
    for cls_id in CLASS_IDS:
        dice, iou, tp, fp, fn = dice_iou(pred == cls_id, target == cls_id)
        row[f"dice_class_{cls_id}"] = dice
        row[f"iou_class_{cls_id}"] = iou
        row[f"bf1_1px_class_{cls_id}"] = boundary_f1(pred == cls_id, target == cls_id, 1)
        row[f"bf1_3px_class_{cls_id}"] = boundary_f1(pred == cls_id, target == cls_id, 3)
        row[f"hd95_class_{cls_id}"] = hd95(pred == cls_id, target == cls_id)
        row[f"pred_area_ratio_class_{cls_id}"] = float((pred == cls_id).mean())
        row[f"gt_area_ratio_class_{cls_id}"] = float((target == cls_id).mean())
        dices.append(dice); ious.append(iou)
        bf1_1.append(row[f"bf1_1px_class_{cls_id}"])
        bf1_3.append(row[f"bf1_3px_class_{cls_id}"])
        hd95s.append(row[f"hd95_class_{cls_id}"])
    row["dice_fg_macro"] = float(np.mean(dices))
    row["iou_fg_macro"] = float(np.mean(ious))
    row["bf1_1px_fg_macro"] = float(np.mean(bf1_1))
    row["bf1_3px_fg_macro"] = float(np.mean(bf1_3))
    row["hd95_fg_macro"] = float(np.nanmean(hd95s)) if not np.all(np.isnan(hd95s)) else float("nan")
    return row


def load_label(path: Path, color_map: dict[tuple[int, int, int], int] | None = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        return image.astype(np.uint8)
    if color_map is None:
        raise ValueError(f"{path} is colour-coded; provide --color-predictions or use indexed labels.")
    out = np.zeros(image.shape[:2], dtype=np.uint8)
    matched = np.zeros(image.shape[:2], dtype=bool)
    for color, cls_id in color_map.items():
        hit = np.all(image[:, :, :3] == np.asarray(color, dtype=np.uint8), axis=2)
        out[hit] = cls_id
        matched |= hit
    if not matched.all():
        raise ValueError(f"Unknown colours in {path}")
    return out


def load_prediction(path: Path, color_map: dict[tuple[int, int, int], int] | None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        logits = np.load(path)
        if logits.ndim == 4 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim != 3 or logits.shape[0] != 4:
            raise ValueError(f"Expected logits [4,H,W] in {path}, got {logits.shape}")
        return logits.argmax(axis=0).astype(np.uint8)
    return load_label(path, color_map)


def default_color_map() -> dict[tuple[int, int, int], int]:
    # BGR: matches configs/Glomeruli.py and cv2.imread without RGB conversion.
    return {(0, 0, 0): 0, (46, 254, 100): 1, (0, 0, 255): 2, (255, 0, 0): 3,
            (255, 0, 170): 0, (88, 250, 244): 0}


def run(pred_dir: Path, gt_dir: Path, output_dir: Path, resolution: int | None, color_predictions: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_size = None if resolution is None else (resolution, resolution)
    color_map = default_color_map()
    rows, matrix = [], np.zeros((4, 4), dtype=np.int64)
    for gt_path in sorted(gt_dir.glob("*.png")):
        pred_path = pred_dir / gt_path.name
        npy_path = pred_dir / f"{gt_path.stem}.npy"
        if not pred_path.exists() and not npy_path.exists():
            raise FileNotFoundError(f"Missing prediction for {gt_path.name}")
        pred = load_prediction(npy_path if npy_path.exists() else pred_path, color_map if color_predictions else None)
        target = load_label(gt_path, color_map)
        pred, target = resize_label(pred, target_size), resize_label(target, target_size)
        rows.append(evaluate_pair(pred, target, gt_path.name))
        matrix += confusion_matrix(pred, target)
    per_image = pd.DataFrame(rows)
    per_image.to_csv(output_dir / "metrics_per_image.csv", index=False)
    numeric = per_image.drop(columns=["filename"]).mean(numeric_only=True).to_dict()
    summary = {"num_images": len(rows), "resolution": resolution or "native", **numeric,
               "protocol": "4-class argmax logits; classes 1-3 foreground; nearest label resize"}
    pd.DataFrame([summary]).to_csv(output_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(matrix, index=[f"gt_{CLASS_NAMES[i]}" for i in range(4)], columns=[f"pred_{CLASS_NAMES[i]}" for i in range(4)]).to_csv(output_dir / "confusion_matrix.csv")
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(256, 1024), default=None)
    parser.add_argument("--color-predictions", action="store_true", help="Predicted PNGs use the configured BGR palette.")
    args = parser.parse_args()
    run(args.pred_dir, args.gt_dir, args.output_dir, args.resolution, args.color_predictions)


if __name__ == "__main__":
    main()
