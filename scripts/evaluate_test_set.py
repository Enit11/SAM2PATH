import csv
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch

PROJECT_ROOT = r"C:\Users\ASUS\SAM2PATH"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sam2path_inference import GlomeruliInference


CKPT_PATH = (
    r"C:\Users\ASUS\SAM2PATH\outputs\glomeruli\run\glomeruli_10pct_mt_scfpb_sipp_seed42\Logs"
    r"\20260722-003303\sam2path\version_0\checkpoints\epoch=99-step=17700.ckpt"
)
IMAGE_DIR = r"C:\Users\ASUS\SAM2PATH\dataset\test\image"
MASK_DIR = r"C:\Users\ASUS\SAM2PATH\dataset\test\mask"
OUTPUT_DIR = r"C:\Users\ASUS\SAM2PATH\outputs\glomeruli\evaluation\sipp_test"


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def update_confusion(counts: dict[int, dict[str, int]], pred: np.ndarray, gt: np.ndarray, classes: list[int]) -> None:
    valid = np.isin(gt, classes)
    pred_eval = pred[valid]
    gt_eval = gt[valid]
    for cls_id in classes:
        pred_c = pred_eval == cls_id
        gt_c = gt_eval == cls_id
        counts[cls_id]["tp"] += int(np.logical_and(pred_c, gt_c).sum())
        counts[cls_id]["fp"] += int(np.logical_and(pred_c, ~gt_c).sum())
        counts[cls_id]["fn"] += int(np.logical_and(~pred_c, gt_c).sum())


def summarize_counts(counts: dict[int, dict[str, int]]) -> dict:
    per_class = {}
    total_tp = total_fp = total_fn = 0
    for cls_id, values in counts.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_class[cls_id] = {
            "dice": safe_div(2 * tp, 2 * tp + fp + fn),
            "iou": safe_div(tp, tp + fp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    dices = [values["dice"] for values in per_class.values()]
    ious = [values["iou"] for values in per_class.values()]
    return {
        "dice_micro": safe_div(2 * total_tp, 2 * total_tp + total_fp + total_fn),
        "dice_macro": float(np.mean(dices)) if dices else 0.0,
        "iou_micro": safe_div(total_tp, total_tp + total_fp + total_fn),
        "iou_macro": float(np.mean(ious)) if ious else 0.0,
        "per_class": per_class,
    }


def prediction_from_prob(prob_map: np.ndarray, classes: list[int]) -> np.ndarray:
    valid_probs = prob_map[classes]
    return np.asarray(classes, dtype=np.uint8)[np.argmax(valid_probs, axis=0)]


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inferencer = GlomeruliInference(ckpt_path=CKPT_PATH, project_root=PROJECT_ROOT)
    inferencer.load_model()
    classes = [cls_id for cls_id in inferencer.valid_foreground_classes() if cls_id in (1, 2, 3)]

    raw_counts = {cls_id: {"tp": 0, "fp": 0, "fn": 0} for cls_id in classes}
    refined_counts = {cls_id: {"tp": 0, "fp": 0, "fn": 0} for cls_id in classes}
    rows = []

    files = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".png"))
    for idx, name in enumerate(files, start=1):
        image_path = os.path.join(IMAGE_DIR, name)
        mask_path = os.path.join(MASK_DIR, name)
        if not os.path.exists(mask_path):
            print(f"[WARN] Missing mask for {name}, skipped")
            continue

        image = inferencer.read_rgb_img(image_path)
        mask_raw = inferencer.read_mask(mask_path)
        gt = inferencer.preprocess_label_mask(mask_raw)
        image_tensor = inferencer.preprocess_image(image)
        prob_map = inferencer.predict_probabilities(image_tensor)

        valid_gt = np.isin(gt, classes)
        raw_pred = prediction_from_prob(prob_map, classes)
        raw_pred = raw_pred * valid_gt.astype(np.uint8)
        refined_pred = inferencer.postprocess_prediction(raw_pred, prob_map=prob_map)
        refined_pred = refined_pred * valid_gt.astype(np.uint8)

        image_raw_counts = {cls_id: {"tp": 0, "fp": 0, "fn": 0} for cls_id in classes}
        image_refined_counts = {cls_id: {"tp": 0, "fp": 0, "fn": 0} for cls_id in classes}
        update_confusion(raw_counts, raw_pred, gt, classes)
        update_confusion(refined_counts, refined_pred, gt, classes)
        update_confusion(image_raw_counts, raw_pred, gt, classes)
        update_confusion(image_refined_counts, refined_pred, gt, classes)

        raw_summary = summarize_counts(image_raw_counts)
        refined_summary = summarize_counts(image_refined_counts)
        rows.append(
            {
                "image": name,
                "raw_dice_micro": raw_summary["dice_micro"],
                "raw_dice_macro": raw_summary["dice_macro"],
                "refined_dice_micro": refined_summary["dice_micro"],
                "refined_dice_macro": refined_summary["dice_macro"],
            }
        )
        print(
            f"[{idx:03d}/{len(files):03d}] {name} "
            f"raw Dice={raw_summary['dice_micro']:.4f}, refined Dice={refined_summary['dice_micro']:.4f}"
        )

    raw = summarize_counts(raw_counts)
    refined = summarize_counts(refined_counts)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = os.path.join(OUTPUT_DIR, f"metrics_{timestamp}.txt")
    csv_path = os.path.join(OUTPUT_DIR, f"per_image_{timestamp}.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {CKPT_PATH}\n")
        f.write(f"classes: {classes}\n")
        f.write("evaluation: foreground-only, matching project ignore_index=0 protocol\n\n")
        for title, summary in [("raw", raw), ("instance_refined", refined)]:
            f.write(f"[{title}]\n")
            f.write(f"Dice/micro: {summary['dice_micro']:.6f}\n")
            f.write(f"Dice_Bal/macro: {summary['dice_macro']:.6f}\n")
            f.write(f"IoU/micro: {summary['iou_micro']:.6f}\n")
            f.write(f"IoU_Bal/macro: {summary['iou_macro']:.6f}\n")
            for cls_id, values in summary["per_class"].items():
                f.write(
                    f"class_{cls_id}: Dice={values['dice']:.6f}, IoU={values['iou']:.6f}, "
                    f"TP={values['tp']}, FP={values['fp']}, FN={values['fn']}\n"
                )
            f.write("\n")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "raw_dice_micro",
                "raw_dice_macro",
                "refined_dice_micro",
                "refined_dice_macro",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n[RESULT] raw")
    print(f"Dice/micro: {raw['dice_micro']:.6f}")
    print(f"Dice_Bal/macro: {raw['dice_macro']:.6f}")
    print(f"IoU/micro: {raw['iou_micro']:.6f}")
    print(f"IoU_Bal/macro: {raw['iou_macro']:.6f}")
    print("[RESULT] instance_refined")
    print(f"Dice/micro: {refined['dice_micro']:.6f}")
    print(f"Dice_Bal/macro: {refined['dice_macro']:.6f}")
    print(f"IoU/micro: {refined['iou_micro']:.6f}")
    print(f"IoU_Bal/macro: {refined['iou_macro']:.6f}")
    print(f"[OK] Saved summary: {summary_path}")
    print(f"[OK] Saved per-image CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
