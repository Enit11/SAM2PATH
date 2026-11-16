"""Diagnose same-class instance merges on a fixed subset of a segmentation test set.

For a predicted connected component that overlaps at least two GT connected
components of the same class, report only its GT-background pixels lying close
to a pair of those GT instances.  These pixels are the candidate connection
bridge, not generic false-positive background.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam2path_inference import GlomeruliInference


PALETTE = np.array([[0, 0, 0], [100, 254, 46], [255, 0, 0], [0, 0, 255]], dtype=np.uint8)


def components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Return component count, labels and statistics; discard centroids."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    return count, labels, stats


def distance_to(component: np.ndarray) -> np.ndarray:
    """Euclidean distance from every pixel to a binary component."""
    return cv2.distanceTransform((~component).astype(np.uint8), cv2.DIST_L2, 5)


def make_panel(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, bridge: np.ndarray, margin: np.ndarray) -> np.ndarray:
    gt_rgb = PALETTE[gt]
    pred_rgb = PALETTE[pred]
    # Red: target-class logit exceeds background; blue: background wins.
    normalized = np.clip((margin + 4.0) / 8.0, 0.0, 1.0)
    heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    bridge_overlay = image.copy()
    bridge_overlay[bridge] = np.array([255, 255, 0], dtype=np.uint8)
    return np.concatenate((image, gt_rgb, pred_rgb, heat, bridge_overlay), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs.Glomeruli_uni2_spherical_hrrefine")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bridge-radius", type=float, default=24.0,
                        help="Max distance (1024-space pixels) from each of two GT instances.")
    args = parser.parse_args()

    all_names = sorted(path.name for path in args.image_dir.glob("*.png"))
    if not all_names:
        raise FileNotFoundError(f"No PNG files in {args.image_dir}")
    picked = all_names if args.num_samples >= len(all_names) else sorted(random.Random(args.seed).sample(all_names, args.num_samples))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "panels").mkdir(exist_ok=True)
    (args.output_dir / "selected_images.txt").write_text("\n".join(picked) + "\n", encoding="utf-8")

    inferencer = GlomeruliInference(args.checkpoint, args.project_root, config_module=args.config)
    pl_module = inferencer.load_model()
    # The underlying prompt model returns a Python list (one tensor per image),
    # while the Lightning wrapper stacks that list into [B, C, H, W].  Use the
    # wrapper to exactly match ``predict_probabilities`` used by inference.
    pl_module.eval()
    rows: list[dict[str, object]] = []

    for index, name in enumerate(picked, start=1):
        print(f"[{index}/{len(picked)}] {name}", flush=True)
        image = inferencer.read_rgb_img(str(args.image_dir / name))
        gt = inferencer.preprocess_label_mask(inferencer.read_mask(str(args.mask_dir / name)))
        image_tensor = inferencer.preprocess_image(image).unsqueeze(0).to(inferencer.device)
        with torch.no_grad():
            logits, _ = pl_module(image_tensor)
            probs = torch.softmax(logits, dim=1)[0].float().cpu().numpy()
            logits_np = logits[0].float().cpu().numpy()
        pred = probs.argmax(axis=0).astype(np.uint8)
        image_1024 = cv2.resize(image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        for cls_id in (1, 2, 3):
            _, gt_labels, gt_stats = components(gt == cls_id)
            n_pred, pred_labels, pred_stats = components(pred == cls_id)
            for pred_id in range(1, n_pred):
                pred_component = pred_labels == pred_id
                # Exclude microscopic components: they cannot plausibly merge instances.
                if int(pred_stats[pred_id, cv2.CC_STAT_AREA]) < 6:
                    continue
                touched = np.unique(gt_labels[pred_component])
                touched = [int(v) for v in touched if v > 0]
                if len(touched) < 2:
                    continue
                for pos_a, gt_a in enumerate(touched):
                    instance_a = gt_labels == gt_a
                    d_a = distance_to(instance_a)
                    for gt_b in touched[pos_a + 1:]:
                        instance_b = gt_labels == gt_b
                        d_b = distance_to(instance_b)
                        bridge = pred_component & (gt == 0) & (d_a <= args.bridge_radius) & (d_b <= args.bridge_radius)
                        if not np.any(bridge):
                            continue
                        target_prob = probs[cls_id][bridge]
                        background_prob = probs[0][bridge]
                        logit_margin = (logits_np[cls_id] - logits_np[0])[bridge]
                        row = {
                            "image": name,
                            "class_id": cls_id,
                            "pred_component": pred_id,
                            "gt_instance_a": gt_a,
                            "gt_instance_b": gt_b,
                            "bridge_pixels": int(bridge.sum()),
                            "target_probability_mean": float(target_prob.mean()),
                            "target_probability_median": float(np.median(target_prob)),
                            "background_probability_mean": float(background_prob.mean()),
                            "background_probability_median": float(np.median(background_prob)),
                            "target_minus_background_probability_mean": float((target_prob - background_prob).mean()),
                            "target_minus_background_logit_mean": float(logit_margin.mean()),
                            "target_minus_background_logit_median": float(np.median(logit_margin)),
                        }
                        rows.append(row)
                        panel = make_panel(image_1024, gt, pred, bridge, logits_np[cls_id] - logits_np[0])
                        panel_name = f"{Path(name).stem}_c{cls_id}_p{pred_id}_gt{gt_a}-{gt_b}.png"
                        cv2.imwrite(str(args.output_dir / "panels" / panel_name), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    fieldnames = [
        "image", "class_id", "pred_component", "gt_instance_a", "gt_instance_b", "bridge_pixels",
        "target_probability_mean", "target_probability_median", "background_probability_mean",
        "background_probability_median", "target_minus_background_probability_mean",
        "target_minus_background_logit_mean", "target_minus_background_logit_median",
    ]
    csv_path = args.output_dir / "bridge_statistics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = args.output_dir / "summary.txt"
    summary.write_text(
        f"sampled_images={len(picked)}\nmerged_instance_pairs={len(rows)}\n"
        f"bridge_radius={args.bridge_radius}\n"
        "Each CSV row is a GT-background bridge candidate where one predicted same-class "
        "component overlaps two GT instances.\n",
        encoding="utf-8",
    )
    print(f"[DONE] {len(rows)} merged-instance bridge pairs. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
