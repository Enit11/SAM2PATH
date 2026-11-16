"""
Inference utilities for SAM2PATH glomeruli segmentation.

This module loads a training checkpoint, runs inference on random samples from
the test set, and exports side-by-side comparisons.
"""

import os
import random
import sys
import warnings
import importlib

import albumentations
import cv2
import numpy as np
import torch
import math

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    HAS_CRF = True
except Exception:
    HAS_CRF = False

warnings.filterwarnings("ignore")


class GlomeruliInference:
    PALETTE = {
        0: (0, 0, 0),
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 0, 255),
        4: (170, 0, 255),
    }

    def __init__(
        self,
        ckpt_path: str,
        project_root: str = None,
        device: str = None,
        config_module: str = "configs.Glomeruli",
    ):
        self.ckpt_path = ckpt_path
        self.project_root = project_root or r"C:\Users\ASUS\SAM2PATH"
        device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)

        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)
        os.chdir(self.project_root)

        from image_mask_dataset import convert_color_mask, read_mask, read_rgb_img
        from main import get_metrics, get_model, get_pl_module

        self.cfg = importlib.import_module(config_module).cfg
        self.convert_color_mask = convert_color_mask
        self.read_mask = read_mask
        self.read_rgb_img = read_rgb_img
        self.get_metrics = get_metrics
        self.get_model = get_model
        self.get_pl_module = get_pl_module
        self.model = None

    def load_model(self):
        if self.model is not None:
            return self.model

        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state_dict = state["state_dict"]
        sam_model = self.get_model(self.cfg)
        metrics = self.get_metrics(self.cfg)
        pl_module = self.get_pl_module(self.cfg, model=sam_model, metrics=metrics)

        checkpoint_has_geometry = any(
            key.startswith("model.boundary_head.") or key.startswith("model.distance_head.")
            for key in state_dict
        )
        model_geometry_enabled = getattr(pl_module.model, "geometry_enabled", None)
        if model_geometry_enabled is not None and checkpoint_has_geometry != model_geometry_enabled:
            raise ValueError(
                "Checkpoint and config build different model architectures: "
                f"checkpoint_has_geometry={checkpoint_has_geometry}, "
                f"config_geometry_enabled={model_geometry_enabled}. "
                "Use the configuration that created this checkpoint."
            )

        for key in ("instance_memory", "instance_memory_ready", "instance_memory_ptr"):
            state_dict.pop(key, None)
        if "foundation_prototypes" in state_dict and hasattr(pl_module, "foundation_prototypes"):
            loaded_proto = state_dict["foundation_prototypes"]
            if loaded_proto.ndim == 2 and loaded_proto.shape[0] == pl_module.num_classes + 1:
                pl_module.foundation_prototypes = loaded_proto.new_zeros(loaded_proto.shape)
        if "foundation_proto_ready" in state_dict and hasattr(pl_module, "foundation_proto_ready"):
            loaded_ready = state_dict["foundation_proto_ready"]
            if loaded_ready.ndim == 1 and loaded_ready.shape[0] == pl_module.num_classes + 1:
                pl_module.foundation_proto_ready = loaded_ready.new_zeros(loaded_ready.shape)
        target_state = pl_module.state_dict()
        incompatible = {
            key: (tuple(value.shape), tuple(target_state[key].shape))
            for key, value in state_dict.items()
            if key in target_state and value.shape != target_state[key].shape
        }
        if incompatible:
            print("[INFO] Skipping incompatible checkpoint tensors:")
            for key, (source_shape, target_shape) in incompatible.items():
                print(f"  {key}: checkpoint {source_shape} -> model {target_shape}")
                state_dict.pop(key)
        missing, unexpected = pl_module.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[INFO] Missing model tensors after partial load: {missing}")
        if unexpected:
            print(f"[INFO] Unexpected checkpoint tensors after partial load: {unexpected}")
        pl_module.to(self.device)
        pl_module.eval()

        self.model = pl_module
        return pl_module

    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls_id, color in self.PALETTE.items():
            canvas[mask == cls_id] = color
        return canvas

    def valid_foreground_classes(self) -> list[int]:
        mapped_classes = set()
        mask_color_map = getattr(self.cfg.dataset, "mask_color_map", None)
        if mask_color_map is not None:
            mapped_classes = {int(v) for v in mask_color_map.values() if int(v) > 0}
        if mapped_classes:
            return sorted(mapped_classes)
        return list(range(1, int(self.cfg.dataset.num_classes) + 1))

    def normalize_map_to_uint8(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        arr = arr - arr.min()
        vmax = arr.max()
        if vmax > 0:
            arr = arr / vmax
        return (arr * 255).astype(np.uint8)

    def summarize_feature_map(self, feature_map: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(feature_map), axis=0)

    def make_feature_grid(
        self,
        feature_map: np.ndarray,
        max_channels: int = 16,
        cols: int = 4,
        tile_size: int = 256,
    ) -> np.ndarray:
        c, _, _ = feature_map.shape
        channel_energy = np.var(feature_map.reshape(c, -1), axis=1)
        picked = np.argsort(channel_energy)[::-1][: min(max_channels, c)]

        tiles = []
        for idx in picked:
            vis = self.normalize_map_to_uint8(feature_map[idx])
            vis = cv2.resize(vis, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
            cv2.putText(
                vis,
                f"ch={int(idx)}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            tiles.append(vis)

        if not tiles:
            return np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

        rows = int(math.ceil(len(tiles) / cols))
        blank = np.zeros_like(tiles[0])
        while len(tiles) < rows * cols:
            tiles.append(blank.copy())

        row_images = []
        for row_idx in range(rows):
            row_tiles = tiles[row_idx * cols : (row_idx + 1) * cols]
            row_images.append(np.concatenate(row_tiles, axis=1))
        return np.concatenate(row_images, axis=0)

    def relabel_small_components(
        self,
        mask: np.ndarray,
        prob_map: np.ndarray | None = None,
        min_area_by_class: dict[int, int] | None = None,
        radius: int = 9,
        confidence_threshold: float = 0.82,
        neighbor_ratio_threshold: float = 0.55,
    ) -> np.ndarray:
        """Relabel unreliable islands to the dominant neighboring class."""
        min_area_by_class = min_area_by_class or {cls_id: 180 for cls_id in self.valid_foreground_classes()}
        if 1 in min_area_by_class:
            min_area_by_class[1] = max(min_area_by_class[1], 450)
        out = mask.astype(np.uint8).copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))

        for cls_id, min_area in min_area_by_class.items():
            cls_mask = (out == cls_id).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls_mask, 8)
            for idx in range(1, num_labels):
                area = stats[idx, cv2.CC_STAT_AREA]
                component = labels == idx
                mean_conf = 0.0
                if prob_map is not None and cls_id < prob_map.shape[0]:
                    mean_conf = float(prob_map[cls_id][component].mean())
                is_uncertain = mean_conf < confidence_threshold
                if area >= min_area:
                    continue

                dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
                ring = dilated & ~component
                neighbors = out[ring]
                neighbors = neighbors[neighbors != cls_id]

                foreground_neighbors = neighbors[neighbors > 0]
                if foreground_neighbors.size > 0:
                    counts = np.bincount(foreground_neighbors)
                    replacement = int(counts.argmax())
                    neighbor_ratio = counts[replacement] / foreground_neighbors.size
                    if area >= min_area and neighbor_ratio < neighbor_ratio_threshold:
                        continue
                elif neighbors.size > 0:
                    replacement = int(np.bincount(neighbors).argmax())
                else:
                    replacement = 0
                out[component] = replacement

        return out

    def fill_small_background_holes(
        self,
        mask: np.ndarray,
        max_area: int = 600,
        radius: int = 9,
    ) -> np.ndarray:
        """Fill small background holes using the dominant surrounding foreground class."""
        out = mask.astype(np.uint8).copy()
        background = (out == 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background, 8)

        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area >= max_area:
                continue

            component = labels == idx
            dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
            ring = dilated & ~component
            neighbors = out[ring]
            foreground_neighbors = neighbors[neighbors > 0]
            if foreground_neighbors.size == 0:
                continue

            replacement = int(np.bincount(foreground_neighbors).argmax())
            out[component] = replacement

        return out

    def refine_enclosed_components(
        self,
        mask: np.ndarray,
        max_area_by_class: dict[int, int] | None = None,
        radius: int = 21,
        dominant_ratio: float = 0.62,
        min_foreground_ratio: float = 0.45,
        iterations: int = 2,
    ) -> np.ndarray:
        """Reassign small islands only when they are enclosed by another foreground class."""
        max_area_by_class = max_area_by_class or {
            1: 3500,
            2: 1800,
            3: 1800,
        }
        valid_classes = self.valid_foreground_classes()
        out = mask.astype(np.uint8).copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))

        for _ in range(iterations):
            changed = False
            for cls_id in valid_classes:
                cls_mask = (out == cls_id).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls_mask, 8)
                max_area = max_area_by_class.get(cls_id, 1000)

                for idx in range(1, num_labels):
                    area = stats[idx, cv2.CC_STAT_AREA]
                    if area >= max_area:
                        continue

                    component = labels == idx
                    dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
                    ring = dilated & ~component
                    ring_values = out[ring]
                    if ring_values.size == 0:
                        continue

                    foreground = ring_values[(ring_values > 0) & (ring_values != cls_id)]
                    foreground_ratio = foreground.size / ring_values.size
                    if foreground.size == 0 or foreground_ratio < min_foreground_ratio:
                        continue

                    counts = np.bincount(foreground, minlength=max(valid_classes) + 1)
                    replacement = int(counts.argmax())
                    ratio = counts[replacement] / foreground.size
                    if replacement > 0 and replacement != cls_id and ratio >= dominant_ratio:
                        out[component] = replacement
                        changed = True

            if not changed:
                break

        return out

    def apply_morphology(self, mask: np.ndarray, kernel_size: int = 5, min_area: int = 120) -> np.ndarray:
        out = np.zeros_like(mask, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        for cls_id in self.valid_foreground_classes():
            cls = (mask == cls_id).astype(np.uint8)
            if cls.sum() == 0:
                continue
            cls = cv2.morphologyEx(cls, cv2.MORPH_OPEN, kernel)
            cls = cv2.morphologyEx(cls, cv2.MORPH_CLOSE, kernel)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls, 8)
            cleaned = np.zeros_like(cls)
            for idx in range(1, num_labels):
                area = stats[idx, cv2.CC_STAT_AREA]
                if area >= min_area:
                    cleaned[labels == idx] = 1
            out[cleaned > 0] = cls_id
        return out

    def suppress_invalid_classes(self, mask: np.ndarray) -> np.ndarray:
        valid_classes = set(self.valid_foreground_classes())
        mask = mask.astype(np.uint8).copy()
        mask[~np.isin(mask, list(valid_classes) + [0])] = 0
        return mask

    def uses_background_supervision(self) -> bool:
        semi_cfg = getattr(self.cfg, "semi", None)
        if semi_cfg is not None and bool(semi_cfg.get("background_supervision", False)):
            return True
        return "ignore_index" in self.cfg.dataset and self.cfg.dataset.ignore_index is None

    def geometry_boundary_deadhesion(
        self,
        mask: np.ndarray,
        prob_map: np.ndarray | None = None,
        geometry: dict | None = None,
        boundary_threshold: float = 0.55,
        uncertain_margin_threshold: float = 0.18,
        min_separator_area: int = 18,
    ) -> np.ndarray:
        """Use GASP boundary evidence as a thin separator instead of only visualizing it."""
        if not geometry or "boundary_probability" not in geometry:
            return mask

        boundary = np.asarray(geometry["boundary_probability"], dtype=np.float32)
        if boundary.shape != mask.shape:
            boundary = cv2.resize(boundary, mask.shape[::-1], interpolation=cv2.INTER_LINEAR)

        candidate = boundary >= boundary_threshold
        if prob_map is not None and prob_map.shape[1:] == mask.shape:
            foreground_classes = self.valid_foreground_classes()
            foreground_probs = prob_map[foreground_classes]
            if foreground_probs.shape[0] >= 2:
                top2 = np.partition(foreground_probs, -2, axis=0)[-2:]
                margin = top2[-1] - top2[-2]
                candidate &= margin <= uncertain_margin_threshold

        # Keep separators only where they touch at least two different foreground labels nearby.
        out = mask.astype(np.uint8).copy()
        foreground_classes = self.valid_foreground_classes()
        separator = np.zeros_like(candidate, dtype=bool)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for cls_id in foreground_classes:
            dilated = cv2.dilate((out == cls_id).astype(np.uint8), kernel, iterations=1).astype(bool)
            touches_other = np.zeros_like(candidate, dtype=bool)
            for other_cls in foreground_classes:
                if other_cls == cls_id:
                    continue
                touches_other |= cv2.dilate((out == other_cls).astype(np.uint8), kernel, iterations=1).astype(bool)
            separator |= candidate & dilated & touches_other

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(separator.astype(np.uint8), 8)
        kept = np.zeros_like(separator, dtype=bool)
        for idx in range(1, num_labels):
            if int(stats[idx, cv2.CC_STAT_AREA]) >= min_separator_area:
                kept |= labels == idx

        out[kept] = 0
        return out

    def structure_semantic_refinement(
        self,
        mask: np.ndarray,
        prob_map: np.ndarray | None = None,
        structure_classes: tuple[int, ...] = (1, 2, 3),
        min_area: int = 900,
        dominant_area_ratio: float = 0.68,
        dominant_prob_margin: float = 0.05,
        max_relabel_ratio: float = 0.32,
        max_relabel_area: int = 9000,
        surround_radius: int = 13,
        min_surround_ratio: float = 0.48,
        prob_tolerance: float = 0.08,
    ) -> np.ndarray:
        """Relabel only unreliable minority subregions inside a dominant structure."""
        if prob_map is None:
            return mask

        valid_classes = [cls_id for cls_id in structure_classes if cls_id in self.valid_foreground_classes()]
        if len(valid_classes) < 2:
            return mask

        out = mask.astype(np.uint8).copy()
        foreground = np.isin(out, valid_classes).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        kernel_size = max(1, 2 * surround_radius + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            component = labels == idx
            component_values = out[component]
            class_areas = np.array([(component_values == cls_id).sum() for cls_id in valid_classes], dtype=np.float32)
            if class_areas.sum() <= 0:
                continue

            area_ratio = class_areas / class_areas.sum()
            mean_probs = np.array([prob_map[cls_id][component].mean() for cls_id in valid_classes], dtype=np.float32)
            score = 0.5 * area_ratio + 0.5 * mean_probs
            dominant_idx = int(score.argmax())
            dominant_cls = int(valid_classes[dominant_idx])

            sorted_score = np.sort(score)
            score_margin = float(sorted_score[-1] - sorted_score[-2]) if len(sorted_score) > 1 else 1.0
            is_reliable_instance = (
                float(area_ratio[dominant_idx]) >= dominant_area_ratio
                or score_margin >= dominant_prob_margin
            )
            if not is_reliable_instance:
                continue

            for cls_id in valid_classes:
                if cls_id == dominant_cls:
                    continue

                minority = (out == cls_id) & component
                if not np.any(minority):
                    continue

                num_minor, minor_labels, minor_stats, _ = cv2.connectedComponentsWithStats(
                    minority.astype(np.uint8),
                    8,
                )
                for minor_idx in range(1, num_minor):
                    minor_area = int(minor_stats[minor_idx, cv2.CC_STAT_AREA])
                    minor_ratio = minor_area / float(area)
                    if minor_area > max_relabel_area and minor_ratio > max_relabel_ratio:
                        continue
                    if minor_ratio > max_relabel_ratio and float(area_ratio[dominant_idx]) < 0.82:
                        continue

                    minor_component = minor_labels == minor_idx
                    dilated = cv2.dilate(minor_component.astype(np.uint8), kernel, iterations=1).astype(bool)
                    ring = dilated & ~minor_component & component
                    if np.any(ring):
                        ring_values = out[ring]
                        foreground_ring = ring_values[np.isin(ring_values, valid_classes)]
                        dominant_surround_ratio = (
                            float((foreground_ring == dominant_cls).sum()) / float(foreground_ring.size)
                            if foreground_ring.size > 0
                            else 0.0
                        )
                    else:
                        dominant_surround_ratio = 0.0

                    dominant_prob = float(prob_map[dominant_cls][minor_component].mean())
                    minority_prob = float(prob_map[cls_id][minor_component].mean())
                    probability_allows = dominant_prob + prob_tolerance >= minority_prob
                    context_allows = dominant_surround_ratio >= min_surround_ratio
                    strong_dominant = float(area_ratio[dominant_idx]) >= 0.82 and minor_ratio <= max_relabel_ratio

                    if probability_allows or context_allows or strong_dominant:
                        out[minor_component] = dominant_cls

        return out

    def postprocess_prediction(
        self,
        mask: np.ndarray,
        prob_map: np.ndarray | None = None,
        geometry: dict | None = None,
    ) -> np.ndarray:
        if bool(self.cfg.get("inference", {}).get("preserve_raw_semantics", False)):
            # Preserve annotated background lumina. No hole filling, closing,
            # component relabelling or other morphology in the BSR protocol.
            return mask.copy()
        valid_classes = set(self.valid_foreground_classes())
        if prob_map is not None:
            prob_map = prob_map.copy()
            invalid_classes = [cls_id for cls_id in range(prob_map.shape[0]) if cls_id not in valid_classes and cls_id != 0]
            if invalid_classes:
                prob_map[0] = np.maximum(prob_map[0], prob_map[invalid_classes].max(axis=0))
                prob_map[invalid_classes] = 0
        mask = self.suppress_invalid_classes(mask)
        mask = self.relabel_small_components(mask)
        mask = self.refine_enclosed_components(mask)
        mask = self.apply_morphology(mask)
        mask = self.fill_small_background_holes(mask)
        mask = self.refine_enclosed_components(
            mask,
            max_area_by_class={
                1: 2500,
                2: 1400,
                3: 1400,
            },
            radius=17,
            dominant_ratio=0.58,
            min_foreground_ratio=0.4,
            iterations=2,
        )
        mask = self.relabel_small_components(
            mask,
            prob_map=prob_map,
            min_area_by_class={
                1: 900,
                2: 500,
                3: 500,
            },
            confidence_threshold=0.88,
            neighbor_ratio_threshold=0.6,
        )
        if self.uses_background_supervision():
            mask = self.geometry_boundary_deadhesion(mask, prob_map=prob_map, geometry=geometry)
        mask[~np.isin(mask, list(valid_classes) + [0])] = 0
        return mask

    def apply_crf(self, image: np.ndarray, prob_map: np.ndarray, iterations: int = 5) -> np.ndarray:
        if not HAS_CRF:
            return prob_map
        h, w = prob_map.shape[1:]
        softmax = prob_map.astype(np.float32)
        unary = unary_from_softmax(softmax)
        d = dcrf.DenseCRF2D(w, h, softmax.shape[0])
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image.astype(np.uint8), compat=5)
        Q = d.inference(iterations)
        return np.array(Q, dtype=np.float32).reshape((softmax.shape[0], h, w))

    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        color = self.colorize_mask(mask)
        return cv2.addWeighted(image, 1.0 - alpha, color, alpha, 0.0)

    def overlay_color_mask(
        self, image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45
    ) -> np.ndarray:
        return cv2.addWeighted(image, 1.0 - alpha, color_mask, alpha, 0.0)

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        aug = albumentations.Compose(
            [
                albumentations.Resize(self.cfg.dataset.image_hw[1], self.cfg.dataset.image_hw[0]),
                albumentations.Normalize(
                    mean=self.cfg.dataset.dataset_mean,
                    std=self.cfg.dataset.dataset_std,
                ),
            ]
        )
        out = aug(image=image)
        arr = out["image"].transpose(2, 0, 1)
        return torch.from_numpy(arr).float()

    def preprocess_label_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = self.convert_color_mask(
                mask,
                self.cfg.dataset.mask_color_map,
                self.cfg.dataset.mask_color_mode,
            )
        mask = cv2.resize(
            mask,
            self.cfg.dataset.image_hw[::-1],
            interpolation=cv2.INTER_NEAREST,
        )
        return mask.astype(np.uint8)

    def preprocess_color_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim == 3:
            mask = self.convert_color_mask(
                mask,
                self.cfg.dataset.mask_color_map,
                self.cfg.dataset.mask_color_mode,
            )
        color_mask = self.colorize_mask(mask)
        color_mask = cv2.resize(
            color_mask,
            self.cfg.dataset.image_hw[::-1],
            interpolation=cv2.INTER_NEAREST,
        )
        return color_mask.astype(np.uint8)

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor, gt_mask_tensor: torch.Tensor):
        model = self.load_model()
        images = image_tensor.unsqueeze(0).to(self.device)
        gt_masks = gt_mask_tensor.unsqueeze(0).to(self.device)
        gt_masks, ignored_masks = model.process_masks(gt_masks)
        pred_masks, _ = model(images)
        pred_mask, gt_mask = model.predict_mask(pred_masks, gt_masks, ignored_masks)
        return pred_mask.squeeze(0).cpu().numpy(), gt_mask.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def predict_probabilities(self, image_tensor: torch.Tensor) -> np.ndarray:
        model = self.load_model()
        images = image_tensor.unsqueeze(0).to(self.device)
        pred_masks, _ = model(images)
        probs = torch.softmax(pred_masks, dim=1)
        return probs.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def predict_geometry(self, image_tensor: torch.Tensor) -> dict:
        """Export learned geometry heads for analysis when a GASP model provides them."""
        pl_module = self.load_model()
        prompt_model = pl_module.model
        if not getattr(prompt_model, "geometry_enabled", False):
            return {}
        images = image_tensor.unsqueeze(0).to(self.device)
        _, _, geometry = prompt_model(images, return_aux=True)
        return {
            "boundary_probability": torch.sigmoid(geometry["boundary_logits"]).squeeze().float().cpu().numpy(),
            "distance_maps": geometry["distance_maps"].squeeze(0).float().cpu().numpy(),
        }

    @torch.no_grad()
    def capture_decoder_features(self, image_tensor: torch.Tensor) -> dict:
        pl_module = self.load_model()
        prompt_model = pl_module.model
        mask_decoder = prompt_model.model.mask_decoder
        images = image_tensor.unsqueeze(0).to(self.device)

        captured = {}
        hooks = []

        def save_tensor(name):
            def _hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captured[name] = tensor.detach().float().cpu()

            return _hook

        hooks.append(mask_decoder.output_upscaling[0].register_forward_hook(save_tensor("decoder_stage1")))
        hooks.append(mask_decoder.output_upscaling[1].register_forward_hook(save_tensor("decoder_stage2")))
        hooks.append(mask_decoder.register_forward_hook(save_tensor("low_res_logits")))

        try:
            pred_masks, _ = prompt_model(images)
        finally:
            for hook in hooks:
                hook.remove()

        return {
            "decoder_stage1": captured["decoder_stage1"].squeeze(0).numpy(),
            "decoder_stage2": captured["decoder_stage2"].squeeze(0).numpy(),
            "low_res_logits": captured["low_res_logits"].squeeze(0).numpy(),
            "final_logits": pred_masks[0].detach().float().cpu().numpy(),
        }

    def infer_image(self, image_path: str, mask_path: str) -> dict:
        image = self.read_rgb_img(image_path)
        gt_mask_raw = self.read_mask(mask_path)
        gt_mask_label = self.preprocess_label_mask(gt_mask_raw)
        gt_mask_color = self.preprocess_color_mask(gt_mask_raw)
        image_resized = cv2.resize(
            image,
            self.cfg.dataset.image_hw[::-1],
            interpolation=cv2.INTER_LINEAR,
        )

        image_tensor = self.preprocess_image(image)
        gt_mask_tensor = torch.from_numpy(gt_mask_label).long()
        prob_map = self.predict_probabilities(image_tensor)
        geometry = self.predict_geometry(image_tensor)
        valid_classes = self.valid_foreground_classes()
        if self.uses_background_supervision():
            pred_mask = np.argmax(prob_map, axis=0).astype(np.uint8)
        else:
            pred_mask = np.zeros(prob_map.shape[1:], dtype=np.uint8)
            if valid_classes:
                valid_probs = prob_map[valid_classes]
                pred_mask = np.asarray(valid_classes, dtype=np.uint8)[np.argmax(valid_probs, axis=0)]
        pred_mask = self.suppress_invalid_classes(pred_mask)
        if not self.uses_background_supervision():
            pred_mask = pred_mask * (gt_mask_label > 0).astype(np.uint8)
        pred_mask_pp = self.postprocess_prediction(pred_mask, prob_map=prob_map, geometry=geometry)
        gt_mask_eval = gt_mask_label
        max_prob = prob_map.max(axis=0)
        max_prob_vis = (np.clip(max_prob, 0, 1) * 255).astype(np.uint8)

        return {
            "original_image": image_resized,
            "raw_prediction": pred_mask,
            "prediction": pred_mask_pp,
            "gt_mask": gt_mask_eval,
            "gt_color": gt_mask_color,
            "raw_pred_color": self.colorize_mask(pred_mask),
            "pred_color": self.colorize_mask(pred_mask_pp),
            "gt_overlay": self.overlay_color_mask(image_resized, gt_mask_color),
            "raw_pred_overlay": self.overlay_mask(image_resized, pred_mask),
            "pred_overlay": self.overlay_mask(image_resized, pred_mask_pp),
            "prob_map": prob_map,
            "max_prob_vis": max_prob_vis,
            "geometry": geometry,
        }

    def create_comparison_image(self, image_name: str, infer_result: dict) -> np.ndarray:
        img = infer_result["original_image"]

        if "raw_pred_color" in infer_result:
            top = np.concatenate(
                [img, infer_result["gt_color"], infer_result["raw_pred_color"], infer_result["pred_color"]],
                axis=1,
            )
            bottom = np.concatenate(
                [img, infer_result["gt_overlay"], infer_result["raw_pred_overlay"], infer_result["pred_overlay"]],
                axis=1,
            )
        else:
            top = np.concatenate(
                [img, infer_result["gt_color"], infer_result["pred_color"]],
                axis=1,
            )
            bottom = np.concatenate(
                [img, infer_result["gt_overlay"], infer_result["pred_overlay"]],
                axis=1,
            )
        prob_vis = cv2.applyColorMap(infer_result["max_prob_vis"], cv2.COLORMAP_JET)
        prob_vis = cv2.cvtColor(prob_vis, cv2.COLOR_BGR2RGB)
        row_width = top.shape[1]
        prob_row = np.concatenate([img, prob_vis], axis=1)
        if prob_row.shape[1] < row_width:
            pad = np.zeros((img.shape[0], row_width - prob_row.shape[1], 3), dtype=np.uint8)
            prob_row = np.concatenate([prob_row, pad], axis=1)
        canvas = np.concatenate([top, bottom, prob_row], axis=0)

        cv2.putText(
            canvas,
            f"{image_name} | Original",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            "GT Color Mask",
            (img.shape[1] + 20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            "Raw Prediction" if "raw_pred_color" in infer_result else "Prediction",
            (img.shape[1] * 2 + 20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        if "raw_pred_color" in infer_result:
            cv2.putText(
                canvas,
                "Boundary-Deadhesion",
                (img.shape[1] * 3 + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
        cv2.putText(
            canvas,
            "Max Prob",
            (img.shape[1] + 20, img.shape[0] * 2 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        return canvas

    def save_probability_maps(self, output_dir: str, image_name: str, infer_result: dict) -> None:
        stem = os.path.splitext(image_name)[0]
        prob_dir = os.path.join(output_dir, f"{stem}_probabilities")
        os.makedirs(prob_dir, exist_ok=True)

        max_prob_path = os.path.join(prob_dir, "max_probability.png")
        cv2.imwrite(max_prob_path, infer_result["max_prob_vis"])

        prob_map = infer_result["prob_map"]
        for cls_id in range(prob_map.shape[0]):
            cls_prob = (np.clip(prob_map[cls_id], 0, 1) * 255).astype(np.uint8)
            out_path = os.path.join(prob_dir, f"class_{cls_id}_probability.png")
            cv2.imwrite(out_path, cls_prob)

        geometry = infer_result.get("geometry", {})
        if geometry:
            boundary = (np.clip(geometry["boundary_probability"], 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(prob_dir, "predicted_boundary_probability.png"), boundary)
            for cls_idx, distance_map in enumerate(geometry["distance_maps"], start=1):
                distance_vis = ((np.clip(distance_map, -1, 1) + 1.0) * 127.5).astype(np.uint8)
                cv2.imwrite(os.path.join(prob_dir, f"class_{cls_idx}_signed_distance.png"), distance_vis)

    def save_decoder_feature_maps(
        self,
        output_dir: str,
        image_name: str,
        image_rgb: np.ndarray,
        decoder_result: dict,
    ) -> None:
        stem = os.path.splitext(image_name)[0]
        feature_dir = os.path.join(output_dir, f"{stem}_decoder_features")
        os.makedirs(feature_dir, exist_ok=True)

        stage1 = decoder_result["decoder_stage1"]
        stage2 = decoder_result["decoder_stage2"]
        low_res_logits = decoder_result["low_res_logits"]

        stage1_summary = self.normalize_map_to_uint8(self.summarize_feature_map(stage1))
        stage2_summary = self.normalize_map_to_uint8(self.summarize_feature_map(stage2))

        cv2.imwrite(os.path.join(feature_dir, "decoder_stage1_meanabs.png"), stage1_summary)
        cv2.imwrite(os.path.join(feature_dir, "decoder_stage2_meanabs.png"), stage2_summary)

        stage1_grid = self.make_feature_grid(stage1)
        stage2_grid = self.make_feature_grid(stage2)
        cv2.imwrite(
            os.path.join(feature_dir, "decoder_stage1_top16.png"),
            cv2.cvtColor(stage1_grid, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(feature_dir, "decoder_stage2_top16.png"),
            cv2.cvtColor(stage2_grid, cv2.COLOR_RGB2BGR),
        )

        logits_tiles = []
        for cls_id in range(low_res_logits.shape[0]):
            vis = self.normalize_map_to_uint8(np.abs(low_res_logits[cls_id]))
            vis = cv2.resize(vis, (256, 256), interpolation=cv2.INTER_NEAREST)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
            cv2.putText(
                vis,
                f"logit cls={cls_id}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            logits_tiles.append(vis)
        logits_grid = np.concatenate(logits_tiles, axis=1)
        cv2.imwrite(
            os.path.join(feature_dir, "low_res_logits_classes.png"),
            cv2.cvtColor(logits_grid, cv2.COLOR_RGB2BGR),
        )

        image_small = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        stage1_color = cv2.applyColorMap(stage1_summary, cv2.COLORMAP_JET)
        stage2_color = cv2.applyColorMap(stage2_summary, cv2.COLORMAP_JET)
        stage1_color = cv2.cvtColor(stage1_color, cv2.COLOR_BGR2RGB)
        stage2_color = cv2.cvtColor(stage2_color, cv2.COLOR_BGR2RGB)
        stage1_color = cv2.resize(stage1_color, (256, 256), interpolation=cv2.INTER_NEAREST)
        stage2_color = cv2.resize(stage2_color, (256, 256), interpolation=cv2.INTER_NEAREST)
        overview_top = np.concatenate([image_small, stage1_color, stage2_color], axis=1)
        logits_overview = cv2.resize(logits_grid, (256 * 3, 256), interpolation=cv2.INTER_NEAREST)
        overview = np.concatenate([overview_top, logits_overview], axis=0)

        cv2.putText(
            overview,
            "Original",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overview,
            "Decoder Stage1 MeanAbs",
            (266, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overview,
            "Decoder Stage2 MeanAbs",
            (522, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(
            os.path.join(feature_dir, "decoder_feature_overview.png"),
            cv2.cvtColor(overview, cv2.COLOR_RGB2BGR),
        )


def run_inference(
    ckpt_path: str,
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    num_samples: int = 5,
    seed: int = 42,
    project_root: str = None,
    config_module: str = "configs.Glomeruli_uni2_fusion",
) -> bool:
    print("\n" + "=" * 70)
    print("SAM2PATH Glomeruli Segmentation - Inference")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    print("\n[1/4] Validate inputs...")
    if not os.path.exists(ckpt_path):
        print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
        return False
    print(f"  [OK] Checkpoint: {os.path.basename(ckpt_path)}")

    files = [f for f in os.listdir(image_dir) if f.lower().endswith(".png")]
    files = sorted(files)
    print(f"  [OK] Found {len(files)} test images")

    if num_samples <= 0 or num_samples >= len(files):
        picked = files
        selection_note = f"all {len(files)}"
    else:
        # A fixed seed gives a reproducible subset while avoiding the previous
        # misleading behavior where num_samples was ignored and every test
        # image was processed.
        picked = sorted(random.sample(files, num_samples))
        selection_note = f"a reproducible random subset of {len(picked)}/{len(files)}"
    print(f"\n[2/4] Using {selection_note} test images...")
    for i, name in enumerate(picked, 1):
        print(f"  [{i}] {name}")

    print("\n[3/4] Loading model...")
    try:
        inferencer = GlomeruliInference(
            ckpt_path, project_root, config_module=config_module
        )
        inferencer.load_model()
        print("  [OK] Model loaded")
        print(f"  Device: {inferencer.device}")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n[4/4] Running inference...")
    try:
        for i, name in enumerate(picked, 1):
            print(f"  [{i}/{len(picked)}] Processing {name}...")
            image_path = os.path.join(image_dir, name)
            mask_path = os.path.join(mask_dir, name)
            result = inferencer.infer_image(image_path, mask_path)
            canvas = inferencer.create_comparison_image(name, result)
            out_path = os.path.join(output_dir, f"random_{i - 1:02d}_{name}")
            cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            inferencer.save_probability_maps(output_dir, name, result)
            print("      [OK] Saved")

        print("\n" + "=" * 70)
        print("[OK] Inference finished")
        print(f"  Output dir: {output_dir}")
        print(f"  Created {len(picked)} comparison images")
        print("=" * 70 + "\n")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Inference failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_inference(
        ckpt_path=(
            r"C:\Users\ASUS\SAM2PATH\outputs\glomeruli\run\glomeruli_uni2_resume2\Logs"
            r"\20260916-145447\sam2path\version_0\checkpoints\epoch=64-step=10010.ckpt"
        ),
        image_dir=r"C:\Users\ASUS\SAM2PATH\dataset_patient_split_seed42\test\image",
        mask_dir=r"C:\Users\ASUS\SAM2PATH\dataset_patient_split_seed42\test\mask",
        output_dir=r"C:\Users\ASUS\SAM2PATH\vis_compare\glomeruli_uni2_epoch64_random5_gtmask",
        num_samples=5,
        seed=42,
        project_root=r"C:\Users\ASUS\SAM2PATH",
    )
    sys.exit(0 if success else 1)
