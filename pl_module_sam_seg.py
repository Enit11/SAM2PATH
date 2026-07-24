from functools import partial
import copy
import math

import torch
from pytorch_lightning import LightningModule
from torch import nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
import time
import cv2
import numpy as np

from losses import SAMLoss

class SamSeg(LightningModule):

    def __init__(
            self,
            cfg,
            sam_model: nn.Module,
            metrics: MetricCollection,
            num_classes: int,
            focal_cof: float = 20.,
            dice_cof: float = 1.,
            iou_cof: float = 1.,
            ce_cof: float = 0.,
            rc_cof: float = 0.,
            rc_sigma: float = 0.5,
            lr: float = 0.0001,
            weight_decay: float = 0.01,
            lr_steps: list = (10, 20),
            warmup_steps: int = 0,
            ignored_index=None,
            ignore_index=0,
            semi_cfg=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["sam_model", "metrics"]) # 这将自动记录所有通过 __init__ 传入的参数
        self.model = sam_model
        self.model = sam_model
        self.num_classes = num_classes

        self.loss = SAMLoss(focal_cof, dice_cof, ce_cof, iou_cof, rc_cof=rc_cof, rc_sigma=rc_sigma)

        self.train_metrics = metrics.clone(postfix='/train')
        self.eval_metrics = metrics.clone(postfix='/eval')
        self.test_metrics = metrics.clone(postfix='/test')
        self.eval_metric_updated = False

        self.lr = lr

        self.ignored_index = ignored_index
        self.ignore_index = ignore_index
        self.semi_cfg = semi_cfg or {}
        self.semi_enabled = bool(self.semi_cfg.get("enabled", False))
        self.mt_conf_threshold = float(self.semi_cfg.get("conf_threshold", 0.8))
        self.mt_max_coef = float(self.semi_cfg.get("consistency_coef", 1.0))
        self.mt_warmup_epochs = int(self.semi_cfg.get("warmup_epochs", 10))
        self.mt_use_mse = self.semi_cfg.get("loss_type", "mse").lower() == "mse"
        self.ema_decay = float(self.semi_cfg.get("ema_decay", 0.999))
        self.proto_cfg = self.semi_cfg.get("prototype", {})
        self.proto_enabled = self.semi_enabled and bool(self.proto_cfg.get("enabled", False))
        self.proto_warmup_epochs = int(self.proto_cfg.get("warmup_epochs", 5))
        self.proto_momentum = float(self.proto_cfg.get("momentum", 0.99))
        self.proto_conf_threshold = float(self.proto_cfg.get("conf_threshold", 0.85))
        self.proto_temperature = float(self.proto_cfg.get("temperature", 0.1))
        self.proto_region_coef = float(self.proto_cfg.get("region_proto_coef", 0.05))
        self.proto_intra_coef = float(self.proto_cfg.get("intra_region_coef", 0.02))
        self.proto_sep_coef = float(self.proto_cfg.get("sep_coef", 0.0))
        self.proto_min_region_pixels = int(self.proto_cfg.get("min_region_pixels", 6))
        self.proto_min_region_purity = float(self.proto_cfg.get("min_region_purity", 0.7))
        self.proto_max_regions_per_image = int(self.proto_cfg.get("max_regions_per_image", 16))
        self.proto_max_pixels_per_region = int(self.proto_cfg.get("max_pixels_per_region", 128))
        self.proto_use_unlabeled = bool(self.proto_cfg.get("use_unlabeled", True))
        self.sipp_cfg = self.proto_cfg.get("sipp", {})
        self.sipp_enabled = self.proto_enabled and bool(self.sipp_cfg.get("enabled", False))
        self.sipp_warmup_epochs = int(self.sipp_cfg.get("warmup_epochs", self.proto_warmup_epochs))
        self.sipp_classes = tuple(int(cls_id) for cls_id in self.sipp_cfg.get("classes", [1, 2, 3]))
        self.sipp_classes = tuple(cls_id for cls_id in self.sipp_classes if 1 <= cls_id <= self.num_classes)
        self.sipp_conf_threshold = float(self.sipp_cfg.get("conf_threshold", 0.9))
        self.sipp_min_region_pixels = int(self.sipp_cfg.get("min_region_pixels", 32))
        self.sipp_max_regions_per_image = int(self.sipp_cfg.get("max_regions_per_image", 16))
        self.sipp_boundary_width = int(self.sipp_cfg.get("boundary_width", 3))
        self.sipp_instance_proto_coef = float(self.sipp_cfg.get("instance_proto_coef", 0.02))
        self.sipp_purity_coef = float(self.sipp_cfg.get("purity_coef", 0.01))
        self.sipp_broadcast_coef = float(self.sipp_cfg.get("broadcast_coef", 0.005))
        self.sipp_unlabeled_weight = float(self.sipp_cfg.get("unlabeled_weight", 0.5))

        self.time_and_cnt = [0., 0]
        self.teacher_model = None
        if self.semi_enabled:
            self.teacher_model = copy.deepcopy(self.model)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        if self.proto_enabled:
            self.register_buffer("foundation_prototypes", torch.zeros(self.num_classes + 1, 0), persistent=True)
            self.register_buffer("foundation_proto_ready", torch.zeros(self.num_classes + 1, dtype=torch.bool), persistent=True)

    def forward(self, images):
        # use forward for inference/predictions
        pred_masks, iou_predictions = self.model(images)

        # pred_masks and iou_predictions are lists  将list 变成 torch 张量
        pred_masks = torch.stack(pred_masks, dim=0)
        iou_predictions = torch.stack(iou_predictions, dim=0)

        return pred_masks, iou_predictions

    def forward_with_feature(self, images):
        pred_masks, iou_predictions, features = self.model(images, return_feature=True)
        pred_masks = torch.stack(pred_masks, dim=0)
        iou_predictions = torch.stack(iou_predictions, dim=0)
        features = self.normalize_feature_map(features)
        return pred_masks, iou_predictions, features

    @torch.no_grad()
    def teacher_forward(self, images):
        pred_masks, iou_predictions = self.teacher_model(images)
        pred_masks = torch.stack(pred_masks, dim=0)
        iou_predictions = torch.stack(iou_predictions, dim=0)
        return pred_masks, iou_predictions

    @torch.no_grad()
    def teacher_forward_with_feature(self, images):
        pred_masks, iou_predictions, features = self.teacher_model(images, return_feature=True)
        pred_masks = torch.stack(pred_masks, dim=0)
        iou_predictions = torch.stack(iou_predictions, dim=0)
        features = self.normalize_feature_map(features)
        return pred_masks, iou_predictions, features

    def normalize_feature_map(self, features):
        if features.ndim != 4:
            raise ValueError(f"Expected 4D feature map, got shape {tuple(features.shape)}")
        # Some SAM/SAM2 encoders return NHWC, while the prototype loss expects NCHW.
        if features.shape[1] == features.shape[2] and features.shape[-1] > features.shape[1]:
            features = features.permute(0, 3, 1, 2).contiguous()
        return features

    def calc_loss(self, pred_masks, gt_masks, iou_predictions, ignored_masks, images=None):
        loss_dict = self.loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks, images=images)
        assert "loss" in loss_dict
        return loss_dict

    @torch.no_grad()
    def process_masks(self, gt_masks):
        # gt_cls_masks = [gt_masks == i for i in range(0, self.num_classes + 1)]

        ignored_masks = gt_masks == self.ignore_index
        # gt_cls_masks = torch.stack(gt_cls_masks[1:], dim=1).float()
        ignored_masks = ignored_masks.unsqueeze(1).long()
        return gt_masks, ignored_masks

    def predict_mask(self, pred_masks, gt_masks, ignored_masks):
        # pred_masks = [batch_size, #classes, h, w]
        # note class 0 is always for ignored classes
        pred_masks = torch.argmax(pred_masks[:, 1:, ...], dim=1) + 1
        pred_masks = pred_masks * (1 - ignored_masks.squeeze(1))

        if self.ignored_index is not None:
            pred_masks[pred_masks == self.ignored_index] = 0
            gt_masks[gt_masks == self.ignored_index] = 0

        return pred_masks, gt_masks

    def mt_coef(self):
        if not self.semi_enabled:
            return 0.0
        if self.mt_warmup_epochs <= 0:
            return self.mt_max_coef
        progress = min(float(self.current_epoch + 1) / float(self.mt_warmup_epochs), 1.0)
        return self.mt_max_coef * math.exp(-5.0 * (1.0 - progress) ** 2)

    def consistency_loss(self, student_logits, teacher_logits):
        student_probs = torch.softmax(student_logits, dim=1)
        teacher_probs = torch.softmax(teacher_logits, dim=1)
        confidence = teacher_probs.max(dim=1).values
        valid = confidence >= self.mt_conf_threshold
        if not torch.any(valid):
            return student_logits.new_tensor(0.0)

        if self.mt_use_mse:
            diff = (student_probs - teacher_probs) ** 2
            diff = diff.mean(dim=1)
            return diff[valid].mean()

        student_log_probs = torch.log_softmax(student_logits, dim=1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
        return kl[valid].mean()

    def proto_active(self):
        return self.proto_enabled and self.current_epoch >= self.proto_warmup_epochs

    def sipp_active(self):
        return self.sipp_enabled and self.current_epoch >= self.sipp_warmup_epochs and len(self.sipp_classes) >= 2

    def ensure_prototypes(self, feature_dim, device):
        if not self.proto_enabled:
            return
        if self.foundation_prototypes.numel() == 0:
            self.foundation_prototypes = torch.zeros(self.num_classes + 1, feature_dim, device=device)
            self.foundation_proto_ready = torch.zeros(self.num_classes + 1, dtype=torch.bool, device=device)

    def resize_labels_for_feature(self, labels, feature_hw):
        labels = labels.unsqueeze(1).float()
        labels = F.interpolate(labels, size=feature_hw, mode="nearest")
        return labels.squeeze(1).long()

    def resize_conf_for_feature(self, confidence, feature_hw):
        confidence = confidence.unsqueeze(1).float()
        confidence = F.interpolate(confidence, size=feature_hw, mode="nearest")
        return confidence.squeeze(1)

    def iter_regions(self, label_map, valid_map=None):
        label_np = label_map.detach().cpu().numpy().astype(np.int32)
        if valid_map is None:
            valid_np = np.ones_like(label_np, dtype=bool)
        else:
            valid_np = valid_map.detach().cpu().numpy().astype(bool)

        regions = []
        for cls_id in range(1, self.num_classes + 1):
            cls_mask = (label_np == cls_id) & valid_np
            if cls_mask.sum() < self.proto_min_region_pixels:
                continue
            num_labels, components = cv2.connectedComponents(cls_mask.astype(np.uint8), connectivity=8)
            for comp_id in range(1, num_labels):
                comp = components == comp_id
                area = int(comp.sum())
                if area < self.proto_min_region_pixels:
                    continue
                purity = float((label_np[comp] == cls_id).mean()) if area > 0 else 0.0
                if purity < self.proto_min_region_purity:
                    continue
                regions.append((area, cls_id, comp))

        regions.sort(key=lambda item: item[0], reverse=True)
        return regions[: self.proto_max_regions_per_image]

    @torch.no_grad()
    def update_foundation_prototypes(self, teacher_features, labels, confidence=None):
        if not self.proto_enabled:
            return teacher_features.new_tensor(0.0)

        self.ensure_prototypes(teacher_features.shape[1], teacher_features.device)
        feature_hw = teacher_features.shape[-2:]
        labels_small = self.resize_labels_for_feature(labels, feature_hw)
        valid_small = labels_small > 0
        if confidence is not None:
            conf_small = self.resize_conf_for_feature(confidence, feature_hw)
            valid_small = valid_small & (conf_small >= self.proto_conf_threshold)

        updates = 0
        for b_idx in range(teacher_features.shape[0]):
            regions = self.iter_regions(labels_small[b_idx], valid_small[b_idx])
            for _, cls_id, comp_np in regions:
                comp = torch.from_numpy(comp_np).to(device=teacher_features.device, dtype=torch.bool)
                region_feature = teacher_features[b_idx, :, comp].mean(dim=1)
                region_feature = F.normalize(region_feature, dim=0)
                if self.foundation_proto_ready[cls_id]:
                    updated = (
                        self.proto_momentum * self.foundation_prototypes[cls_id]
                        + (1.0 - self.proto_momentum) * region_feature
                    )
                else:
                    updated = region_feature
                    self.foundation_proto_ready[cls_id] = True
                self.foundation_prototypes[cls_id] = F.normalize(updated, dim=0)
                updates += 1

        return teacher_features.new_tensor(float(updates))

    def region_prototype_losses(self, student_features, labels, confidence=None):
        if not self.proto_enabled:
            zero = student_features.new_tensor(0.0)
            return zero, zero, zero

        self.ensure_prototypes(student_features.shape[1], student_features.device)
        ready_classes = torch.where(self.foundation_proto_ready[1:])[0] + 1
        if ready_classes.numel() < 2:
            zero = student_features.new_tensor(0.0)
            return zero, zero, zero

        feature_hw = student_features.shape[-2:]
        labels_small = self.resize_labels_for_feature(labels, feature_hw)
        valid_small = labels_small > 0
        if confidence is not None:
            conf_small = self.resize_conf_for_feature(confidence, feature_hw)
            valid_small = valid_small & (conf_small >= self.proto_conf_threshold)

        proto_bank = self.foundation_prototypes[ready_classes].detach()
        proto_bank = F.normalize(proto_bank, dim=1)
        region_losses = []
        intra_losses = []
        device = student_features.device

        for b_idx in range(student_features.shape[0]):
            regions = self.iter_regions(labels_small[b_idx], valid_small[b_idx])
            for _, cls_id, comp_np in regions:
                target_positions = torch.where(ready_classes == cls_id)[0]
                if target_positions.numel() == 0:
                    continue

                comp = torch.from_numpy(comp_np).to(device=device, dtype=torch.bool)
                pixel_features = student_features[b_idx, :, comp].transpose(0, 1)
                if pixel_features.shape[0] > self.proto_max_pixels_per_region:
                    pick = torch.randperm(pixel_features.shape[0], device=device)[: self.proto_max_pixels_per_region]
                    pixel_features = pixel_features[pick]

                pixel_features = F.normalize(pixel_features, dim=1)
                region_feature = F.normalize(pixel_features.mean(dim=0), dim=0)
                logits = torch.matmul(region_feature.unsqueeze(0), proto_bank.t()) / max(self.proto_temperature, 1e-6)
                target = target_positions[:1].to(device=device)
                region_losses.append(F.cross_entropy(logits, target))

                intra = 1.0 - torch.matmul(pixel_features, region_feature)
                intra_losses.append(intra.mean())

        if not region_losses:
            zero = student_features.new_tensor(0.0)
            return zero, zero, self.prototype_separation_loss(ready_classes)

        region_loss = torch.stack(region_losses).mean()
        intra_loss = torch.stack(intra_losses).mean()
        sep_loss = self.prototype_separation_loss(ready_classes)
        return region_loss, intra_loss, sep_loss

    def prototype_separation_loss(self, ready_classes):
        if self.proto_sep_coef <= 0.0 or ready_classes.numel() < 2:
            return self.foundation_prototypes.new_tensor(0.0)

        prototypes = F.normalize(self.foundation_prototypes[ready_classes].detach(), dim=1)
        sim = torch.matmul(prototypes, prototypes.t())
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        sim = sim[~eye]
        return F.relu(sim - 0.2).mean()

    def iter_sipp_regions(self, label_map, valid_map=None):
        label_np = label_map.detach().cpu().numpy().astype(np.int32)
        if valid_map is None:
            valid_np = np.ones_like(label_np, dtype=bool)
        else:
            valid_np = valid_map.detach().cpu().numpy().astype(bool)

        regions = []
        for cls_id in self.sipp_classes:
            cls_mask = (label_np == cls_id) & valid_np
            if cls_mask.sum() < self.sipp_min_region_pixels:
                continue
            num_labels, components, stats, _ = cv2.connectedComponentsWithStats(cls_mask.astype(np.uint8), 8)
            for comp_id in range(1, num_labels):
                area = int(stats[comp_id, cv2.CC_STAT_AREA])
                if area >= self.sipp_min_region_pixels:
                    regions.append((area, int(cls_id), components == comp_id))

        regions.sort(key=lambda item: item[0], reverse=True)
        return regions[: self.sipp_max_regions_per_image]

    def make_interior_mask(self, comp_np):
        if self.sipp_boundary_width <= 0:
            return comp_np
        kernel_size = 2 * self.sipp_boundary_width + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        interior = cv2.erode(comp_np.astype(np.uint8), kernel, iterations=1).astype(bool)
        if interior.sum() < self.sipp_min_region_pixels:
            return comp_np
        return interior

    def sipp_losses(self, student_logits, student_features, labels, confidence=None):
        zero = student_logits.new_tensor(0.0)
        if not self.sipp_active() or student_features is None:
            return zero, zero, zero, zero

        self.ensure_prototypes(student_features.shape[1], student_features.device)
        ready_classes = torch.tensor(self.sipp_classes, device=student_logits.device, dtype=torch.long)
        ready_mask = self.foundation_proto_ready[ready_classes].to(device=student_logits.device)
        ready_classes = ready_classes[ready_mask]
        if ready_classes.numel() < 2:
            return zero, zero, zero, zero

        proto_bank = self.foundation_prototypes[ready_classes].detach()
        proto_bank = F.normalize(proto_bank, dim=1)
        probs = torch.softmax(student_logits, dim=1)
        feature_hw = student_features.shape[-2:]
        valid_full = labels > 0
        if confidence is not None:
            conf_full = confidence >= self.sipp_conf_threshold
            valid_full = valid_full & conf_full

        instance_proto_losses = []
        purity_losses = []
        broadcast_losses = []
        region_count = 0
        device = student_logits.device
        sipp_class_tensor = torch.tensor(self.sipp_classes, device=device, dtype=torch.long)

        for b_idx in range(labels.shape[0]):
            regions = self.iter_sipp_regions(labels[b_idx], valid_full[b_idx])
            for _, cls_id, comp_np in regions:
                target_positions = torch.where(ready_classes == cls_id)[0]
                if target_positions.numel() == 0:
                    continue

                comp_full = torch.from_numpy(comp_np).to(device=device, dtype=torch.bool)
                if not torch.any(comp_full):
                    continue

                full_region_probs = probs[b_idx, sipp_class_tensor][:, comp_full]
                if full_region_probs.numel() == 0:
                    continue
                region_distribution = full_region_probs.mean(dim=1)
                region_distribution = region_distribution / region_distribution.sum().clamp_min(1e-6)
                purity_losses.append(-(region_distribution * torch.log(region_distribution.clamp_min(1e-6))).sum())

                interior_np = self.make_interior_mask(comp_np)
                interior = torch.from_numpy(interior_np).to(device=device, dtype=torch.bool)
                if torch.any(interior):
                    interior_logits = student_logits[b_idx, :, interior].transpose(0, 1)
                    targets = torch.full((interior_logits.shape[0],), cls_id, device=device, dtype=torch.long)
                    broadcast_losses.append(F.cross_entropy(interior_logits, targets))

                comp_small = F.interpolate(
                    comp_full.unsqueeze(0).unsqueeze(0).float(),
                    size=feature_hw,
                    mode="nearest",
                ).squeeze(0).squeeze(0).bool()
                if torch.any(comp_small):
                    instance_token = student_features[b_idx, :, comp_small].mean(dim=1)
                    instance_token = F.normalize(instance_token, dim=0)
                    logits = torch.matmul(instance_token.unsqueeze(0), proto_bank.t()) / max(self.proto_temperature, 1e-6)
                    instance_proto_losses.append(F.cross_entropy(logits, target_positions[:1].to(device=device)))

                region_count += 1

        if not instance_proto_losses and not purity_losses and not broadcast_losses:
            return zero, zero, zero, zero

        instance_proto_loss = torch.stack(instance_proto_losses).mean() if instance_proto_losses else zero
        purity_loss = torch.stack(purity_losses).mean() if purity_losses else zero
        broadcast_loss = torch.stack(broadcast_losses).mean() if broadcast_losses else zero
        return instance_proto_loss, purity_loss, broadcast_loss, student_logits.new_tensor(float(region_count))

    def training_step(self, batch, batch_idx):
        if self.semi_enabled and isinstance(batch, dict):
            (images_l, gt_masks) = batch["labeled"]
            images_w, images_s = batch["unlabeled"]
            gt_masks, ignored_masks = self.process_masks(gt_masks)

            if self.proto_enabled:
                pred_masks_l, iou_predictions_l, student_features_l = self.forward_with_feature(images_l)
            else:
                pred_masks_l, iou_predictions_l = self(images_l)
                student_features_l = None
            losses = self.calc_loss(pred_masks_l, gt_masks, iou_predictions_l, ignored_masks=ignored_masks, images=images_l)

            with torch.no_grad():
                if self.proto_enabled:
                    teacher_logits, _, teacher_features_w = self.teacher_forward_with_feature(images_w)
                    _, _, teacher_features_l = self.teacher_forward_with_feature(images_l)
                else:
                    teacher_logits, _ = self.teacher_forward(images_w)
                    teacher_features_w = None
                    teacher_features_l = None
            if self.proto_enabled:
                student_logits, _, student_features_s = self.forward_with_feature(images_s)
            else:
                student_logits, _ = self(images_s)
                student_features_s = None
            mt_loss = self.consistency_loss(student_logits, teacher_logits)
            coef = self.mt_coef()

            proto_region_loss = pred_masks_l.new_tensor(0.0)
            proto_intra_loss = pred_masks_l.new_tensor(0.0)
            proto_sep_loss = pred_masks_l.new_tensor(0.0)
            sipp_instance_loss = pred_masks_l.new_tensor(0.0)
            sipp_purity_loss = pred_masks_l.new_tensor(0.0)
            sipp_broadcast_loss = pred_masks_l.new_tensor(0.0)
            sipp_regions = pred_masks_l.new_tensor(0.0)
            proto_updates = pred_masks_l.new_tensor(0.0)
            if self.proto_active():
                with torch.no_grad():
                    proto_updates = proto_updates + self.update_foundation_prototypes(teacher_features_l, gt_masks)
                    if self.proto_use_unlabeled:
                        teacher_probs = torch.softmax(teacher_logits, dim=1)
                        teacher_conf, teacher_pseudo = teacher_probs.max(dim=1)
                        teacher_pseudo = teacher_pseudo * (teacher_conf >= self.proto_conf_threshold).long()
                        proto_updates = proto_updates + self.update_foundation_prototypes(
                            teacher_features_w,
                            teacher_pseudo,
                            confidence=teacher_conf,
                        )

                proto_region_l, proto_intra_l, proto_sep_loss = self.region_prototype_losses(student_features_l, gt_masks)
                if self.proto_use_unlabeled:
                    proto_region_u, proto_intra_u, proto_sep_u = self.region_prototype_losses(
                        student_features_s,
                        teacher_pseudo,
                        confidence=teacher_conf,
                    )
                    proto_region_loss = 0.5 * (proto_region_l + proto_region_u)
                    proto_intra_loss = 0.5 * (proto_intra_l + proto_intra_u)
                    proto_sep_loss = 0.5 * (proto_sep_loss + proto_sep_u)
                else:
                    proto_region_loss = proto_region_l
                    proto_intra_loss = proto_intra_l

                sipp_instance_l, sipp_purity_l, sipp_broadcast_l, sipp_regions_l = self.sipp_losses(
                    pred_masks_l,
                    student_features_l,
                    gt_masks,
                )
                sipp_instance_loss = sipp_instance_l
                sipp_purity_loss = sipp_purity_l
                sipp_broadcast_loss = sipp_broadcast_l
                sipp_regions = sipp_regions_l
                if self.proto_use_unlabeled:
                    sipp_instance_u, sipp_purity_u, sipp_broadcast_u, sipp_regions_u = self.sipp_losses(
                        student_logits,
                        student_features_s,
                        teacher_pseudo,
                        confidence=teacher_conf,
                    )
                    sipp_instance_loss = sipp_instance_l + self.sipp_unlabeled_weight * sipp_instance_u
                    sipp_purity_loss = sipp_purity_l + self.sipp_unlabeled_weight * sipp_purity_u
                    sipp_broadcast_loss = sipp_broadcast_l + self.sipp_unlabeled_weight * sipp_broadcast_u
                    sipp_regions = sipp_regions_l + sipp_regions_u

            proto_total = (
                self.proto_region_coef * proto_region_loss
                + self.proto_intra_coef * proto_intra_loss
                + self.proto_sep_coef * proto_sep_loss
                + self.sipp_instance_proto_coef * sipp_instance_loss
                + self.sipp_purity_coef * sipp_purity_loss
                + self.sipp_broadcast_coef * sipp_broadcast_loss
            )
            total_loss = losses["loss"] + coef * mt_loss + proto_total

            self.log_losses(losses, "train")
            self.log("Loss/train_mt", mt_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_mt_coef", torch.tensor(coef, device=self.device), on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_proto_region", proto_region_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_proto_intra", proto_intra_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_proto_sep", proto_sep_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_sipp_instance", sipp_instance_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_sipp_purity", sipp_purity_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_sipp_broadcast", sipp_broadcast_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_proto_total", proto_total, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Prototype/updates", proto_updates, on_epoch=True, on_step=True, sync_dist=True)
            self.log("SIPP/regions", sipp_regions, on_epoch=True, on_step=True, sync_dist=True)
            if self.proto_enabled:
                self.log("Prototype/ready_classes", self.foundation_proto_ready[1:].float().sum(), on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_total", total_loss, on_epoch=True, on_step=True, sync_dist=True)

            mask_cls_pred, gt_masks = self.predict_mask(pred_masks_l, gt_masks, ignored_masks=ignored_masks)
            self.train_metrics.update(mask_cls_pred, gt_masks)
            self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
            return total_loss

        images, gt_masks = batch
        gt_masks, ignored_masks = self.process_masks(gt_masks)

        pred_masks, iou_predictions = self(images)
        losses = self.calc_loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks, images=images)

        self.log_losses(losses, "train")

        mask_cls_pred, gt_masks = self.predict_mask(pred_masks, gt_masks, ignored_masks=ignored_masks)
        self.train_metrics.update(mask_cls_pred, gt_masks)
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)

        return losses["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.semi_enabled or self.teacher_model is None:
            return
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.model.parameters()):
                teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1.0 - self.ema_decay)

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        images, gt_masks = batch
        gt_masks, ignored_masks = self.process_masks(gt_masks)

        pred_masks, iou_predictions = self(images)
        losses = self.calc_loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks, images=images)

        mask_cls_pred, gt_masks = self.predict_mask(pred_masks, gt_masks, ignored_masks=ignored_masks)

        if not self.trainer.sanity_checking:
            self.log_losses(losses, "eval")
            self.eval_metrics.update(mask_cls_pred, gt_masks)
            self.eval_metric_updated = True

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            if self.eval_metric_updated:
                self.log_dict(self.eval_metrics.compute(), add_dataloader_idx=False)
            self.eval_metrics.reset()
            self.eval_metric_updated = False

    def test_step(self, batch, batch_idx):
        images, gt_masks = batch
        gt_masks, ignored_masks = self.process_masks(gt_masks)

        pred_masks, iou_predictions = self(images)
        losses = self.calc_loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks, images=images)
        mask_cls_pred, gt_masks = self.predict_mask(pred_masks, gt_masks, ignored_masks=ignored_masks)

        self.log_losses(losses, "test")
        self.test_metrics.update(mask_cls_pred, gt_masks)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), add_dataloader_idx=False)
        self.test_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        images, gt_masks = batch
        gt_masks, ignored_masks = self.process_masks(gt_masks)


        # pred_masks, iou_predictions = self(images)
        with torch.no_grad():
            time_start = time.perf_counter()
            pred_masks, iou_predictions = self.model(images)
            time_predict = time.perf_counter() - time_start

        pred_masks = torch.stack(pred_masks, dim=0)
        iou_predictions = torch.stack(iou_predictions, dim=0)

        self.time_and_cnt[0] += time_predict
        self.time_and_cnt[1] += 1
        print("Average prediction time: %f" % (self.time_and_cnt[0] / self.time_and_cnt[1]))

        mask_cls_pred, gt_masks = self.predict_mask(pred_masks, gt_masks, ignored_masks=ignored_masks)
        return mask_cls_pred

    def log_losses(self, losses, prefiex):
        if prefiex == "train":
            for t in losses:
                self.log("Loss/%s_%s" % (prefiex, t), losses[t], on_epoch=True, on_step=True, sync_dist=True)
        else:
            for t in losses:
                self.log("Loss/%s_%s" % (prefiex, t), losses[t], on_epoch=True, on_step=False, sync_dist=True,
                         add_dataloader_idx=False)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.hparams.lr_steps, verbose=False)
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            elif step < self.hparams.lr_steps[0]:
                return 1.0
            elif step < self.hparams.lr_steps[1]:
                return 0.1
            else:
                return 0.01
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }#[optimizer], [scheduler]
