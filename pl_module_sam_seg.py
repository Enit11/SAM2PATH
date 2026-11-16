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
from network.hyperbolic import exp0, log0, distance2

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
            class_weights=None,
            focal_gamma: float = 2.0,
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
        self.strict_loading = False
        self.model = sam_model
        self.model = sam_model
        self.num_classes = num_classes

        self.loss = SAMLoss(
            focal_cof,
            dice_cof,
            ce_cof,
            iou_cof,
            rc_cof=rc_cof,
            rc_sigma=rc_sigma,
            ignore_index=ignore_index,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )

        self.train_metrics = metrics.clone(postfix='/train')
        self.eval_metrics = metrics.clone(postfix='/eval')
        self.test_metrics = metrics.clone(postfix='/test')
        self.eval_metric_updated = False

        self.lr = lr

        self.ignored_index = ignored_index
        self.ignore_index = ignore_index
        self.semi_cfg = semi_cfg or {}
        probe_cfg = cfg.get("decoder_only_probe", {})
        self.decoder_only_probe = bool(probe_cfg.get("enabled", False))
        adapter_probe_cfg = cfg.get("adapter_decoder_probe", {})
        self.adapter_decoder_probe = bool(adapter_probe_cfg.get("enabled", False))
        if self.decoder_only_probe:
            self.freeze_except_mask_decoder()
        if self.adapter_decoder_probe:
            self.freeze_except_adapter_and_mask_decoder()
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
        self.proto_sep_margin = float(self.proto_cfg.get("sep_margin", 0.2))
        self.proto_geometry = self.proto_cfg.get("geometry", "cosine")
        self.proto_curvature = float(self.proto_cfg.get("curvature", 1.0))
        self.proto_tangent_scale = float(self.proto_cfg.get("tangent_scale", 0.25))
        self.proto_hyperbolic_margin = float(self.proto_cfg.get("hyperbolic_margin", 1.0))
        self.proto_instance_gate_enabled = bool(self.proto_cfg.get("instance_gate_enabled", False))
        self.proto_instance_gate = nn.Linear(256, 1) if self.proto_instance_gate_enabled else nn.Identity()
        self.proto_instance_gate_coef = float(self.proto_cfg.get("instance_gate_coef", 0.0))
        if self.proto_geometry not in ("cosine", "poincare", "dual") or self.proto_curvature <= 0 or self.proto_tangent_scale <= 0:
            raise ValueError("Invalid prototype geometry/curvature/tangent_scale")
        self.proto_min_region_pixels = int(self.proto_cfg.get("min_region_pixels", 6))
        self.proto_min_region_purity = float(self.proto_cfg.get("min_region_purity", 0.7))
        self.proto_max_regions_per_image = int(self.proto_cfg.get("max_regions_per_image", 16))
        self.proto_max_pixels_per_region = int(self.proto_cfg.get("max_pixels_per_region", 128))
        self.proto_use_unlabeled = bool(self.proto_cfg.get("use_unlabeled", True))
        self.proto_unlabeled_weight = float(self.proto_cfg.get("unlabeled_weight", 0.5))
        self.memory_cfg = self.proto_cfg.get("memory", {})
        self.memory_enabled = self.proto_enabled and bool(self.memory_cfg.get("enabled", False))
        self.memory_warmup_epochs = int(self.memory_cfg.get("warmup_epochs", self.proto_warmup_epochs))
        self.memory_size_per_class = int(self.memory_cfg.get("size_per_class", 16))
        self.memory_momentum = float(self.memory_cfg.get("momentum", 0.9))
        self.memory_temperature = float(self.memory_cfg.get("temperature", self.proto_temperature))
        self.memory_logits_coef = float(self.memory_cfg.get("logits_coef", self.memory_cfg.get("consistency_coef", 0.0)))
        self.memory_feature_coef = float(self.memory_cfg.get("feature_coef", self.memory_cfg.get("consistency_coef", 0.0)))
        self.memory_unlabeled_weight = float(self.memory_cfg.get("unlabeled_weight", 0.5))
        self.memory_use_unlabeled = bool(self.memory_cfg.get("use_unlabeled", self.proto_use_unlabeled))

        # Learn which nearby foreground instance owns a boundary pixel. This is
        # intentionally separate from ICP's class-level core-to-boundary path.
        self.geometry_cfg = self.semi_cfg.get("geometry", {})
        self.bsr_cfg = self.geometry_cfg.get("bsr", {})
        self.bsr_enabled = bool(self.bsr_cfg.get("enabled", False))
        self.bsr_ce_coef = float(self.bsr_cfg.get("weighted_ce_coef", 0.1))
        self.geometry_enabled = bool(self.geometry_cfg.get("enabled", False))
        self.geometry_warmup_epochs = int(self.geometry_cfg.get("warmup_epochs", 0))
        self.geometry_boundary_coef = float(self.geometry_cfg.get("boundary_coef", 0.0))
        self.geometry_distance_coef = float(self.geometry_cfg.get("distance_coef", 0.0))
        self.geometry_hard_ce_coef = float(self.geometry_cfg.get("hard_ce_coef", 0.0))
        self.geometry_unsup_coef = float(self.geometry_cfg.get("unsup_coef", 0.0))
        self.geometry_boundary_consistency_coef = float(self.geometry_cfg.get("boundary_consistency_coef", 0.0))
        self.geometry_boundary_width = int(self.geometry_cfg.get("boundary_width", 3))
        self.geometry_sdf_truncation = float(self.geometry_cfg.get("sdf_truncation", 32.0))
        self.geometry_entropy_threshold = float(self.geometry_cfg.get("entropy_threshold", 0.45))
        self.geometry_margin_threshold = float(self.geometry_cfg.get("margin_threshold", 2.0))
        self.geometry_teacher_conf_threshold = float(self.geometry_cfg.get("teacher_conf_threshold", self.mt_conf_threshold))
        self.geometry_vessel_boundary_weight = float(self.geometry_cfg.get("vessel_boundary_weight", 1.0))
        # Supervise same-class inter-instance gaps with labelled GT only.  The
        # diagnosed failure is a teacher/student merge, so pseudo-labels must
        # not be used to fabricate this target.
        self.geometry_separator_enabled = bool(self.geometry_cfg.get("separator_enabled", False))
        self.geometry_separator_radius = int(self.geometry_cfg.get("separator_radius", 6))
        self.geometry_separator_min_instance_pixels = int(self.geometry_cfg.get("separator_min_instance_pixels", 6))
        self.geometry_separator_head_coef = float(self.geometry_cfg.get("separator_head_coef", 0.0))
        self.geometry_separator_logit_coef = float(self.geometry_cfg.get("separator_logit_coef", 0.0))

        self.time_and_cnt = [0., 0]
        self.dual_space = None
        if self.proto_enabled and self.proto_geometry == "dual":
            from network.dual_space_prototype import DualSpaceProjection, PairedPrototypeMemory
            dual_cfg = self.proto_cfg.get("dual_space", {})
            dim = int(dual_cfg.get("dim", 128))
            self.model.dual_space_projection = DualSpaceProjection(256, dim)
            self.dual_space = PairedPrototypeMemory(self.num_classes, dim, self.memory_size_per_class, dual_cfg)
        self.teacher_model = None
        if self.semi_enabled:
            self.teacher_model = copy.deepcopy(self.model)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        if self.proto_enabled:
            self.register_buffer("foundation_prototypes", torch.zeros(self.num_classes + 1, 0), persistent=True)
            self.register_buffer("foundation_proto_ready", torch.zeros(self.num_classes + 1, dtype=torch.bool), persistent=True)
        if self.memory_enabled:
            self.register_buffer("instance_memory", torch.zeros(self.num_classes + 1, self.memory_size_per_class, 0), persistent=True)
            self.register_buffer("instance_memory_ready", torch.zeros(self.num_classes + 1, self.memory_size_per_class, dtype=torch.bool), persistent=True)
            self.register_buffer("instance_memory_ptr", torch.zeros(self.num_classes + 1, dtype=torch.long), persistent=True)
        if self.geometry_enabled and not getattr(self.model, "geometry_enabled", False):
            raise ValueError("semi.geometry.enabled requires model.geometry.enabled and geometry heads.")

    def freeze_except_mask_decoder(self):
        """Make the probe a decoder-only classifier-boundary adaptation."""
        trainable = []
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = name.startswith("model.mask_decoder.")
            if parameter.requires_grad:
                trainable.append(name)
        if not trainable:
            raise RuntimeError("decoder_only_probe found no model.mask_decoder parameters")
        print(f"[decoder_only_probe] Training {len(trainable)} mask-decoder parameter tensors only.")

    def freeze_except_adapter_and_mask_decoder(self):
        """Train the structure adapter and mask decoder while freezing the rest."""
        trainable = []
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = (
                name.startswith("structure_feature_adapter.")
                or name.startswith("model.mask_decoder.")
            )
            if parameter.requires_grad:
                trainable.append(name)
        if not trainable:
            raise RuntimeError("adapter_decoder_probe found no adapter or mask-decoder parameters")
        print(f"[adapter_decoder_probe] Training {len(trainable)} adapter/mask-decoder parameter tensors only.")

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint.get("state_dict", {})

        # Only discard memory when this configuration does not define it. A true
        # Lightning resume must retain its FIFO bank rather than silently reset it.
        if not self.memory_enabled:
            for key in ("instance_memory", "instance_memory_ready", "instance_memory_ptr"):
                state_dict.pop(key, None)
        elif "instance_memory" in state_dict:
            loaded_memory = state_dict["instance_memory"]
            expected_prefix = (self.num_classes + 1, self.memory_size_per_class)
            if loaded_memory.ndim == 3 and tuple(loaded_memory.shape[:2]) == expected_prefix:
                # The feature dimension is lazy because it comes from the decoder.
                # Materialize it before Lightning copies the resumed FIFO bank.
                self.instance_memory = loaded_memory.new_zeros(loaded_memory.shape)
                ready_key = "instance_memory_ready"
                pointer_key = "instance_memory_ptr"
                if ready_key in state_dict:
                    loaded_ready = state_dict[ready_key]
                    if tuple(loaded_ready.shape) == expected_prefix:
                        self.instance_memory_ready = loaded_ready.new_zeros(loaded_ready.shape)
                if pointer_key in state_dict:
                    loaded_pointer = state_dict[pointer_key]
                    if tuple(loaded_pointer.shape) == (self.num_classes + 1,):
                        self.instance_memory_ptr = loaded_pointer.new_zeros(loaded_pointer.shape)
            else:
                raise ValueError(
                    "Checkpoint instance memory is incompatible with this configuration: "
                    f"checkpoint={tuple(loaded_memory.shape)}, "
                    f"expected_prefix={expected_prefix}."
                )

        proto_key = "foundation_prototypes"
        ready_key = "foundation_proto_ready"
        if self.proto_enabled and proto_key in state_dict:
            loaded_proto = state_dict[proto_key]
            if loaded_proto.ndim == 2 and loaded_proto.shape[0] == self.num_classes + 1:
                self.foundation_prototypes = loaded_proto.new_zeros(loaded_proto.shape)
        if self.proto_enabled and ready_key in state_dict:
            loaded_ready = state_dict[ready_key]
            if loaded_ready.ndim == 1 and loaded_ready.shape[0] == self.num_classes + 1:
                    self.foundation_proto_ready = loaded_ready.new_zeros(loaded_ready.shape)

    def on_save_checkpoint(self, checkpoint):
        """Keep checkpoints bounded for the duplicated Mean-Teacher model.

        The teacher is an EMA copy of the student and can be reconstructed at
        startup.  Optimizer moments are also intentionally omitted from these
        research checkpoints: training is restarted through ``init_ckpt_path``
        rather than an exact optimizer resume.  Keeping both 758M-parameter
        networks plus AdamW moments caused host-memory exhaustion while
        serializing the epoch-0 checkpoint.
        """
        state_dict = checkpoint.get("state_dict")
        if state_dict is not None:
            for key in list(state_dict.keys()):
                if key.startswith("teacher_model."):
                    state_dict.pop(key, None)
        # This training workflow uses init_ckpt_path for warm starts.  Do not
        # serialize the very large AdamW moment tensors in every epoch file.
        checkpoint.pop("optimizer_states", None)
        checkpoint.pop("lr_schedulers", None)

    def train(self, mode=True):
        super().train(mode)
        if self.teacher_model is not None:
            self.teacher_model.eval()
        # Frozen foundation models should not enable stochastic training layers.
        sam = getattr(self.model, "model", None)
        uni = getattr(self.model, "extra_encoder", None)
        for module in (getattr(sam, "image_encoder", None), getattr(uni, "model", None)):
            if module is not None and not any(p.requires_grad for p in module.parameters()):
                module.eval()
        return self

    def on_fit_start(self):
        if self.dual_space is not None and self.trainer.world_size > 1:
            raise ValueError('Paired region memory currently supports single-device training only; distributed region synchronization is not implemented.')
        if self.bsr_enabled and not bool(getattr(self.model, 'geometry_cfg', {}).get('bsr_enabled', False)):
            # Small control-flow test doubles have no geometry_cfg; real models do.
            if hasattr(self.model, 'geometry_cfg'):
                raise ValueError('BSR loss requires model.geometry.bsr_enabled=True')

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

    def forward_with_geometry(self, images):
        pred_masks, iou_predictions, features, geometry = self.model(images, return_feature=True, return_aux=True)
        return (
            torch.stack(pred_masks, dim=0),
            torch.stack(iou_predictions, dim=0),
            self.normalize_feature_map(features),
            geometry,
        )

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

    @torch.no_grad()
    def teacher_forward_with_geometry(self, images):
        pred_masks, iou_predictions, features, geometry = self.teacher_model(
            images, return_feature=True, return_aux=True
        )
        return (
            torch.stack(pred_masks, dim=0),
            torch.stack(iou_predictions, dim=0),
            self.normalize_feature_map(features),
            geometry,
        )

    def normalize_feature_map(self, features):
        if features.ndim != 4:
            raise ValueError(f"Expected 4D feature map, got shape {tuple(features.shape)}")
        # Some SAM/SAM2 encoders return NHWC, while the prototype loss expects NCHW.
        if features.shape[1] == features.shape[2] and features.shape[-1] > features.shape[1]:
            features = features.permute(0, 3, 1, 2).contiguous()
        return features

    def calc_loss(self, pred_masks, gt_masks, iou_predictions, ignored_masks, images=None):
        expected_channels = self.num_classes + 1
        if pred_masks.ndim != 4 or pred_masks.shape[1] != expected_channels:
            raise ValueError(
                f"Model output shape {tuple(pred_masks.shape)} is incompatible with "
                f"num_classes={self.num_classes}: expected [B, {expected_channels}, H, W]."
            )
        loss_dict = self.loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks, images=images)
        assert "loss" in loss_dict
        return loss_dict

    @torch.no_grad()
    def process_masks(self, gt_masks):
        # gt_cls_masks = [gt_masks == i for i in range(0, self.num_classes + 1)]
        ignored_masks = (
            torch.zeros_like(gt_masks, dtype=torch.bool)
            if self.ignore_index is None
            else gt_masks == self.ignore_index
        )
        # gt_cls_masks = torch.stack(gt_cls_masks[1:], dim=1).float()
        ignored_masks = ignored_masks.unsqueeze(1).long()
        return gt_masks, ignored_masks

    def predict_mask(self, pred_masks, gt_masks, ignored_masks):
        # pred_masks = [batch_size, #classes, h, w]
        if self.ignore_index is None:
            pred_masks = torch.argmax(pred_masks, dim=1)
        else:
            # Legacy foreground-only configurations reserve class 0 for ignored pixels.
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

    def memory_active(self):
        return self.memory_enabled and self.current_epoch >= self.memory_warmup_epochs

    def geometry_active(self):
        return self.geometry_enabled and self.current_epoch >= self.geometry_warmup_epochs

    def ensure_prototypes(self, feature_dim, device):
        if not self.proto_enabled:
            return
        if self.foundation_prototypes.numel() == 0:
            self.foundation_prototypes = torch.zeros(self.num_classes + 1, feature_dim, device=device)
            self.foundation_proto_ready = torch.zeros(self.num_classes + 1, dtype=torch.bool, device=device)

    def ensure_instance_memory(self, feature_dim, device):
        if not self.memory_enabled:
            return
        if self.instance_memory.numel() == 0:
            self.instance_memory = torch.zeros(
                self.num_classes + 1, self.memory_size_per_class, feature_dim, device=device
            )
            self.instance_memory_ready = torch.zeros(
                self.num_classes + 1, self.memory_size_per_class, dtype=torch.bool, device=device
            )
            self.instance_memory_ptr = torch.zeros(self.num_classes + 1, dtype=torch.long, device=device)

    def iter_instance_regions(self, label_map, min_pixels, max_regions):
        label_np = label_map.detach().cpu().numpy().astype(np.int32)
        regions = []
        for cls_id in range(1, self.num_classes + 1):
            cls_mask = label_np == cls_id
            if int(cls_mask.sum()) < min_pixels:
                continue
            count, components, stats, _ = cv2.connectedComponentsWithStats(cls_mask.astype(np.uint8), 8)
            for comp_id in range(1, count):
                area = int(stats[comp_id, cv2.CC_STAT_AREA])
                if area >= min_pixels:
                    regions.append((area, cls_id, components == comp_id))
        regions.sort(key=lambda item: item[0], reverse=True)
        return regions[:max_regions]

    @staticmethod
    def eroded_core(component, width, min_pixels):
        if width <= 0:
            return component
        kernel_size = 2 * width + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        core = cv2.erode(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        return core if int(core.sum()) >= min_pixels else component

    def feature_token(self, features, batch_index, full_mask):
        small_mask = F.interpolate(
            full_mask.unsqueeze(0).unsqueeze(0).float(), size=features.shape[-2:], mode="nearest"
        ).squeeze(0).squeeze(0).bool()
        if not torch.any(small_mask):
            return None
        return F.normalize(features[batch_index, :, small_mask].mean(dim=1), dim=0)

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
    def update_foundation_prototypes(self, teacher_features, labels, confidence=None, valid_map=None):
        if self.dual_space is not None:
            return self.dual_space.update(self, teacher_features, labels, confidence, valid_map)
        if not self.proto_enabled:
            return teacher_features.new_tensor(0.0)

        self.ensure_prototypes(teacher_features.shape[1], teacher_features.device)
        feature_hw = teacher_features.shape[-2:]
        labels_small = self.resize_labels_for_feature(labels, feature_hw)
        valid_small = labels_small > 0
        if valid_map is not None:
            valid_small = valid_small & self.resize_labels_for_feature(valid_map.long(), feature_hw).bool()
        if confidence is not None:
            conf_small = self.resize_conf_for_feature(confidence, feature_hw)
            valid_small = valid_small & (conf_small >= self.proto_conf_threshold)

        updates = 0
        for b_idx in range(teacher_features.shape[0]):
            regions = self.iter_regions(labels_small[b_idx], valid_small[b_idx])
            for _, cls_id, comp_np in regions:
                comp = torch.from_numpy(comp_np).to(device=teacher_features.device, dtype=torch.bool)
                region_pixels = teacher_features[b_idx, :, comp].transpose(0, 1)
                if self.proto_instance_gate_enabled:
                    weights = torch.sigmoid(self.proto_instance_gate(region_pixels)).float()
                    region_feature = (region_pixels.float() * weights).sum(dim=0) / weights.sum().clamp_min(1e-6)
                else:
                    region_feature = region_pixels.mean(dim=0)
                if self.proto_geometry == "poincare":
                    region_feature = exp0(self.proto_tangent_scale * region_feature, self.proto_curvature)
                    if self.foundation_proto_ready[cls_id]:
                        region_feature = exp0(
                            self.proto_momentum * log0(self.foundation_prototypes[cls_id], self.proto_curvature)
                            + (1-self.proto_momentum) * log0(region_feature, self.proto_curvature), self.proto_curvature
                        )
                    self.foundation_prototypes[cls_id] = region_feature
                    self.foundation_proto_ready[cls_id] = True
                    updates += 1
                    continue
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

    def region_prototype_losses(self, student_features, labels, confidence=None, valid_map=None):
        if self.dual_space is not None:
            return self.dual_space.losses(self, student_features, labels, confidence, valid_map)
        if not self.proto_enabled:
            zero = student_features.new_tensor(0.0)
            return zero, zero, zero, zero

        self.ensure_prototypes(student_features.shape[1], student_features.device)
        ready_classes = torch.where(self.foundation_proto_ready[1:])[0] + 1
        if ready_classes.numel() < 2:
            zero = student_features.new_tensor(0.0)
            return zero, zero, zero, zero

        feature_hw = student_features.shape[-2:]
        labels_small = self.resize_labels_for_feature(labels, feature_hw)
        valid_small = labels_small > 0
        if valid_map is not None:
            valid_small = valid_small & self.resize_labels_for_feature(valid_map.long(), feature_hw).bool()
        if confidence is not None:
            conf_small = self.resize_conf_for_feature(confidence, feature_hw)
            valid_small = valid_small & (conf_small >= self.proto_conf_threshold)

        proto_bank = self.foundation_prototypes[ready_classes].detach()
        if self.proto_geometry != "poincare":
            proto_bank = F.normalize(proto_bank, dim=1)
        region_losses = []
        intra_losses = []
        region_count = 0
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

                if self.proto_instance_gate_enabled:
                    weights = torch.sigmoid(self.proto_instance_gate(pixel_features)).float()
                    pooled_feature = (pixel_features.float() * weights).sum(dim=0) / weights.sum().clamp_min(1e-6)
                else:
                    pooled_feature = pixel_features.mean(dim=0)

                if self.proto_geometry == "poincare":
                    region_feature = exp0(self.proto_tangent_scale * pooled_feature, self.proto_curvature)
                    pixel_features = exp0(self.proto_tangent_scale * pixel_features, self.proto_curvature)
                    logits = -distance2(region_feature.unsqueeze(0), proto_bank, self.proto_curvature).unsqueeze(0) / max(self.proto_temperature, 1e-6)
                    intra = distance2(pixel_features, region_feature, self.proto_curvature)
                else:
                    pixel_features = F.normalize(pixel_features, dim=1)
                    region_feature = F.normalize(pooled_feature, dim=0) if self.proto_instance_gate_enabled else F.normalize(pixel_features.mean(dim=0), dim=0)
                    logits = torch.matmul(region_feature.unsqueeze(0), proto_bank.t()) / max(self.proto_temperature, 1e-6)
                    intra = 1.0 - torch.matmul(pixel_features, region_feature)
                target = target_positions[:1].to(device=device)
                region_losses.append(F.cross_entropy(logits, target))

                intra_losses.append(intra.mean())
                region_count += 1

        sep_loss = self.prototype_separation_loss(student_features, labels, ready_classes, valid_map=valid_map)
        count_tensor = student_features.new_tensor(float(region_count))
        if not region_losses:
            zero = student_features.new_tensor(0.0)
            return zero, zero, sep_loss, count_tensor

        region_loss = torch.stack(region_losses).mean()
        intra_loss = torch.stack(intra_losses).mean()
        return region_loss, intra_loss, sep_loss, count_tensor

    def prototype_separation_loss(self, student_features, labels, ready_classes, valid_map=None):
        if self.proto_sep_coef <= 0.0 or ready_classes.numel() < 2:
            return student_features.new_tensor(0.0)

        bank = self.foundation_prototypes[ready_classes].detach()
        if self.proto_geometry != "poincare":
            bank = F.normalize(bank, dim=1)
        labels_small = self.resize_labels_for_feature(labels, student_features.shape[-2:])
        valid_small = labels_small > 0
        if valid_map is not None:
            valid_small = valid_small & self.resize_labels_for_feature(valid_map.long(), student_features.shape[-2:]).bool()
        losses = []
        for batch_index in range(student_features.shape[0]):
            for position, cls_id in enumerate(ready_classes.tolist()):
                mask = (labels_small[batch_index] == cls_id) & valid_small[batch_index]
                if not torch.any(mask):
                    continue
                token = student_features[batch_index, :, mask].mean(dim=1)
                token = exp0(self.proto_tangent_scale * token, self.proto_curvature) if self.proto_geometry == "poincare" else F.normalize(token, dim=0)
                other_bank = bank[torch.arange(bank.shape[0], device=bank.device) != position]
                if other_bank.numel():
                    if self.proto_geometry == "poincare":
                        losses.append(F.relu(self.proto_hyperbolic_margin - distance2(other_bank, token, self.proto_curvature)).mean())
                    else:
                        losses.append(F.relu(torch.matmul(other_bank, token) - self.proto_sep_margin).mean())
        return torch.stack(losses).mean() if losses else student_features.new_tensor(0.0)

    @torch.no_grad()
    def log_prototype_cosines(self):
        if not self.proto_enabled or self.foundation_prototypes.numel() == 0:
            return
        ready = torch.where(self.foundation_proto_ready[1:])[0] + 1
        if ready.numel() < 2:
            return
        bank = self.foundation_prototypes[ready]
        if self.proto_geometry != "poincare":
            bank = F.normalize(bank, dim=1)
        for left in range(ready.numel()):
            for right in range(left + 1, ready.numel()):
                lhs, rhs = int(ready[left]), int(ready[right])
                self.log(
                    f"Hyperbolic/prototype_distance2_{lhs}_{rhs}" if self.proto_geometry == "poincare" else f"Spherical/prototype_cosine_{lhs}_{rhs}",
                    distance2(bank[left], bank[right], self.proto_curvature) if self.proto_geometry == "poincare" else torch.dot(bank[left], bank[right]),
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                )

    @torch.no_grad()
    def update_instance_memory(self, teacher_features, labels, valid_map=None, confidence=None):
        if self.dual_space is not None:
            return self.dual_space.update(self, teacher_features, labels, confidence=confidence, valid_map=valid_map, memory=True)
        if not self.memory_active():
            return teacher_features.new_tensor(0.0)
        self.ensure_instance_memory(teacher_features.shape[1], teacher_features.device)
        if valid_map is None:
            valid_map = torch.ones_like(labels, dtype=torch.bool)
        else:
            valid_map = valid_map.bool()
        updates = 0
        for batch_index in range(labels.shape[0]):
            regions = self.iter_instance_regions(labels[batch_index], self.proto_min_region_pixels, self.proto_max_regions_per_image)
            for _, cls_id, component_np in regions:
                component = torch.from_numpy(component_np).to(teacher_features.device)
                valid_component = valid_map[batch_index] & component.bool()
                if valid_component.sum() < self.proto_min_region_pixels:
                    continue
                token = self.feature_token(teacher_features, batch_index, valid_component)
                if token is None:
                    continue
                slot = int(self.instance_memory_ptr[cls_id].item())
                if self.instance_memory_ready[cls_id, slot]:
                    token = F.normalize(
                        self.memory_momentum * self.instance_memory[cls_id, slot]
                        + (1.0 - self.memory_momentum) * token,
                        dim=0,
                    )
                self.instance_memory[cls_id, slot] = token
                self.instance_memory_ready[cls_id, slot] = True
                self.instance_memory_ptr[cls_id] = (slot + 1) % self.memory_size_per_class
                updates += 1
        return teacher_features.new_tensor(float(updates))

    def structure_aware_memory_loss(self, student_logits, student_features, labels, valid_map=None):
        if self.dual_space is not None:
            match, _, _, count = self.dual_space.losses(self, student_features, labels, valid_map=valid_map, memory=True)
            return match, student_features.sum() * 0, count
        zero = student_logits.new_tensor(0.0)
        if not self.memory_active():
            return zero, zero, zero
        ready = self.instance_memory_ready
        if not torch.any(ready):
            return zero, zero, zero
        feature_losses = []
        contrast_losses = []
        count = 0
        bank_tokens = []
        bank_classes = []
        for cls_id in range(1, self.num_classes + 1):
            class_memory = self.instance_memory[cls_id, ready[cls_id]].detach()
            if class_memory.numel():
                bank_tokens.append(F.normalize(class_memory, dim=1))
                bank_classes.append(torch.full((class_memory.shape[0],), cls_id, device=student_features.device))
        if not bank_tokens:
            return zero, zero, zero
        bank = torch.cat(bank_tokens, dim=0)
        bank_labels = torch.cat(bank_classes, dim=0)
        if valid_map is None:
            valid_map = torch.ones_like(labels, dtype=torch.bool)
        else:
            valid_map = valid_map.bool()
        for batch_index in range(labels.shape[0]):
            regions = self.iter_instance_regions(labels[batch_index], self.proto_min_region_pixels, self.proto_max_regions_per_image)
            for _, cls_id, component_np in regions:
                component = torch.from_numpy(component_np).to(student_features.device)
                valid_component = valid_map[batch_index].to(student_features.device) & component.bool()
                if valid_component.sum() < self.proto_min_region_pixels:
                    continue
                token = self.feature_token(student_features, batch_index, valid_component)
                class_memory = self.instance_memory[cls_id, ready[cls_id]].detach()
                if token is None or class_memory.numel() == 0:
                    continue
                class_memory = F.normalize(class_memory, dim=1)
                feature_losses.append(1.0 - torch.matmul(class_memory, token).max())
                logits = torch.matmul(token.unsqueeze(0), bank.t()) / max(self.memory_temperature, 1e-6)
                target_positions = torch.where(bank_labels == cls_id)[0]
                target_log_prob = torch.logsumexp(logits[:, target_positions], dim=1) - torch.logsumexp(logits, dim=1)
                contrast_losses.append(-target_log_prob.mean())
                count += 1
        return (
            torch.stack(contrast_losses).mean() if contrast_losses else zero,
            torch.stack(feature_losses).mean() if feature_losses else zero,
            student_logits.new_tensor(float(count)),
        )

    def geometry_targets(self, labels):
        """Build semantic boundaries, SDFs, and same-class gap targets from GT.

        A separator pixel is GT background covered by dilation neighbourhoods
        from at least two different connected components of one foreground
        class. This target is generated only for labelled images.
        """
        device = labels.device
        labels_np = labels.detach().cpu().numpy().astype(np.int32)
        batch_size, height, width = labels_np.shape
        boundary = np.zeros((batch_size, height, width), dtype=bool)
        separator = np.zeros((batch_size, height, width), dtype=bool)
        # Labels use 0 for background and 1..num_classes for foreground.
        # Store one distance channel per foreground class; channel cls_id-1
        # corresponds exactly to label cls_id.
        distance_maps = np.zeros((batch_size, self.num_classes, height, width), dtype=np.float32)
        kernel_size = max(1, 2 * self.geometry_boundary_width + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        separator_radius = max(1, self.geometry_separator_radius)
        separator_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * separator_radius + 1, 2 * separator_radius + 1)
        )
        truncation = max(self.geometry_sdf_truncation, 1.0)
        for batch_index, label in enumerate(labels_np):
            valid = label > 0
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                shifted = np.roll(label, shift=(dy, dx), axis=(0, 1))
                difference = (label != shifted) & valid
                if dy == 1:
                    difference[0, :] = False
                elif dy == -1:
                    difference[-1, :] = False
                elif dx == 1:
                    difference[:, 0] = False
                else:
                    difference[:, -1] = False
                boundary[batch_index] |= difference
            boundary[batch_index] = cv2.dilate(boundary[batch_index].astype(np.uint8), kernel, iterations=1).astype(bool)
            for cls_id in range(1, self.num_classes + 1):
                foreground = (label == cls_id).astype(np.uint8)
                if not foreground.any():
                    distance_maps[batch_index, cls_id - 1] = 1.0
                    continue
                if foreground.all():
                    distance_maps[batch_index, cls_id - 1] = -1.0
                    continue
                outside = cv2.distanceTransform(1 - foreground, cv2.DIST_L2, 5)
                inside = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
                signed = np.clip((outside - inside) / truncation, -1.0, 1.0)
                distance_maps[batch_index, cls_id - 1] = signed
                if self.geometry_separator_enabled:
                    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
                        foreground, connectivity=8
                    )
                    coverage = np.zeros((height, width), dtype=np.uint16)
                    for component_id in range(1, component_count):
                        if component_stats[component_id, cv2.CC_STAT_AREA] < self.geometry_separator_min_instance_pixels:
                            continue
                        component = (component_labels == component_id).astype(np.uint8)
                        coverage += cv2.dilate(component, separator_kernel, iterations=1).astype(np.uint16)
                    separator[batch_index] |= (coverage >= 2) & (label == 0)
        return (
            torch.from_numpy(boundary.astype(np.float32)).to(device=device).unsqueeze(1),
            torch.from_numpy(distance_maps).to(device=device),
            torch.from_numpy(separator.astype(np.float32)).to(device=device).unsqueeze(1),
        )

    def supervised_geometry_losses(self, geometry, labels, targets=None):
        zero = labels.new_tensor(0.0, dtype=torch.float32)
        if not self.geometry_active() or not geometry:
            return zero, zero, zero, zero, zero
        boundary_target, distance_target, separator_target = targets if targets is not None else self.geometry_targets(labels)
        boundary_logits = geometry["boundary_logits"]
        distance_prediction = geometry["distance_maps"]
        if distance_prediction.shape != distance_target.shape:
            raise ValueError(f'SDF output/target mismatch: {distance_prediction.shape} vs {distance_target.shape}')
        boundary_weight = torch.ones_like(boundary_target)
        if self.geometry_vessel_boundary_weight != 1.0:
            vessel = (labels == 2).float().unsqueeze(1)
            boundary_weight = boundary_weight + vessel * (self.geometry_vessel_boundary_weight - 1.0)
        positives = boundary_target.sum().clamp_min(1.0)
        negatives = (1.0 - boundary_target).sum().clamp_min(1.0)
        pos_weight = (negatives / positives).clamp(max=50.0)
        boundary_map = F.binary_cross_entropy_with_logits(
            boundary_logits, boundary_target, pos_weight=pos_weight, reduction="none"
        )
        boundary_loss = (boundary_map * boundary_weight).sum() / boundary_weight.sum().clamp_min(1.0)
        distance_loss = F.l1_loss(distance_prediction.float(), distance_target) if self.bsr_enabled else F.smooth_l1_loss(distance_prediction, distance_target)
        separator_head_loss = zero
        if self.geometry_separator_enabled and "separator_logits" in geometry:
            positives = separator_target.sum().clamp_min(1.0)
            negatives = (1.0 - separator_target).sum().clamp_min(1.0)
            pos_weight = (negatives / positives).clamp(max=100.0)
            separator_head_loss = F.binary_cross_entropy_with_logits(
                geometry["separator_logits"], separator_target, pos_weight=pos_weight
            )
        return boundary_loss, distance_loss, separator_head_loss, boundary_target.mean(), separator_target.mean()

    def separator_logit_loss(self, student_logits, separator_target):
        """Directly make GT same-class gaps compete as background in main logits."""
        zero = student_logits.new_tensor(0.0)
        valid = separator_target[:, 0].bool()
        if not self.geometry_separator_enabled or not torch.any(valid):
            return zero
        background_labels = torch.zeros_like(separator_target[:, 0], dtype=torch.long)
        per_pixel = F.cross_entropy(student_logits, background_labels, reduction="none")
        return per_pixel[valid].mean()

    def hard_boundary_losses(self, student_logits, geometry, labels, boundary_target, teacher_logits=None, teacher_geometry=None):
        zero = student_logits.new_tensor(0.0)
        if not self.geometry_active():
            return zero, zero, zero
        probabilities = torch.softmax(student_logits, dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=1)
        top2 = torch.topk(student_logits, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        hard = boundary_target.squeeze(1).bool() & (
            (entropy >= self.geometry_entropy_threshold) | (margin <= self.geometry_margin_threshold)
        )
        hard_supervised = F.cross_entropy(student_logits, labels, reduction="none")
        hard_loss = hard_supervised[hard].mean() if torch.any(hard) else zero
        unsup_loss = zero
        if self.geometry_unsup_coef > 0 and teacher_logits is not None and teacher_geometry is not None and teacher_geometry:
            raise ValueError('Legacy geometry.unsup_coef pairs different images. Set it to 0 and use boundary_consistency_coef.')
        return hard_loss, unsup_loss, hard.float().mean()

    def boundary_consistency_loss(self, student_geometry, teacher_logits, teacher_geometry):
        zero = teacher_logits.new_tensor(0.0)
        if not student_geometry or not teacher_geometry or "boundary_logits" not in student_geometry or "boundary_logits" not in teacher_geometry:
            return zero
        with torch.no_grad():
            probs = torch.softmax(teacher_logits, dim=1)
            conf = probs.max(dim=1).values
            teacher_boundary = torch.sigmoid(teacher_geometry["boundary_logits"])
            valid = (conf >= self.geometry_teacher_conf_threshold) & (teacher_boundary[:, 0] >= 0.5)
        if not torch.any(valid):
            return zero
        student_boundary = torch.sigmoid(student_geometry["boundary_logits"])[:, 0]
        return F.mse_loss(student_boundary[valid], teacher_boundary[:, 0][valid])

    def training_step(self, batch, batch_idx):
        if self.dual_space is not None:
            if self.dual_space.pending:
                raise RuntimeError('Uncommitted paired-memory writes from previous batch')
            self.dual_space.defer_writes = True
        if self.semi_enabled and isinstance(batch, dict):
            (images_l, gt_masks) = batch["labeled"]
            images_w, images_s = batch["unlabeled"]
            gt_masks, ignored_masks = self.process_masks(gt_masks)

            use_geometry = self.geometry_active()
            if use_geometry:
                pred_masks_l, iou_predictions_l, student_features_l, geometry_l = self.forward_with_geometry(images_l)
            elif self.proto_enabled:
                pred_masks_l, iou_predictions_l, student_features_l = self.forward_with_feature(images_l)
                geometry_l = {}
            else:
                pred_masks_l, iou_predictions_l = self(images_l)
                student_features_l = None
                geometry_l = {}
            losses = self.calc_loss(pred_masks_l, gt_masks, iou_predictions_l, ignored_masks=ignored_masks, images=images_l)

            with torch.no_grad():
                if use_geometry:
                    teacher_logits, _, teacher_features_w, teacher_geometry_w = self.teacher_forward_with_geometry(images_w)
                    teacher_logits_l, _, teacher_features_l, _ = self.teacher_forward_with_geometry(images_l)
                elif self.proto_enabled:
                    teacher_logits, _, teacher_features_w = self.teacher_forward_with_feature(images_w)
                    teacher_logits_l, _, teacher_features_l = self.teacher_forward_with_feature(images_l)
                    teacher_geometry_w = {}
                else:
                    teacher_logits, _ = self.teacher_forward(images_w)
                    teacher_logits_l = None
                    teacher_features_w = None
                    teacher_features_l = None
                    teacher_geometry_w = {}
            if use_geometry:
                student_logits, _, student_features_s, geometry_s = self.forward_with_geometry(images_s)
            elif self.proto_enabled:
                student_logits, _, student_features_s = self.forward_with_feature(images_s)
                geometry_s = {}
            else:
                student_logits, _ = self(images_s)
                student_features_s = None
                geometry_s = {}
            mt_loss = self.consistency_loss(student_logits, teacher_logits)
            coef = self.mt_coef()

            proto_region_loss = pred_masks_l.new_tensor(0.0)
            proto_intra_loss = pred_masks_l.new_tensor(0.0)
            proto_sep_loss = pred_masks_l.new_tensor(0.0)
            proto_updates = pred_masks_l.new_tensor(0.0)
            memory_updates = pred_masks_l.new_tensor(0.0)
            memory_logits_loss = pred_masks_l.new_tensor(0.0)
            memory_feature_loss = pred_masks_l.new_tensor(0.0)
            memory_regions = pred_masks_l.new_tensor(0.0)
            proto_updates_labeled = pred_masks_l.new_tensor(0.0)
            proto_updates_unlabeled = pred_masks_l.new_tensor(0.0)
            proto_regions_labeled = pred_masks_l.new_tensor(0.0)
            proto_regions_unlabeled = pred_masks_l.new_tensor(0.0)
            memory_updates_labeled = pred_masks_l.new_tensor(0.0)
            memory_updates_unlabeled = pred_masks_l.new_tensor(0.0)
            memory_regions_labeled = pred_masks_l.new_tensor(0.0)
            memory_regions_unlabeled = pred_masks_l.new_tensor(0.0)
            teacher_conf = None
            teacher_pseudo = None
            teacher_valid = None
            geometry_boundary_loss = pred_masks_l.new_tensor(0.0)
            geometry_distance_loss = pred_masks_l.new_tensor(0.0)
            geometry_hard_loss = pred_masks_l.new_tensor(0.0)
            geometry_unsup_loss = pred_masks_l.new_tensor(0.0)
            geometry_boundary_consistency = pred_masks_l.new_tensor(0.0)
            geometry_separator_head_loss = pred_masks_l.new_tensor(0.0)
            geometry_separator_logit_loss = pred_masks_l.new_tensor(0.0)
            geometry_boundary_density = pred_masks_l.new_tensor(0.0)
            geometry_sdf_magnitude = pred_masks_l.new_tensor(0.0)
            geometry_separator_density = pred_masks_l.new_tensor(0.0)
            geometry_hard_fraction = pred_masks_l.new_tensor(0.0)
            if self.proto_active():
                needs_unlabeled_structure = (
                    self.proto_use_unlabeled or self.memory_use_unlabeled
                )
                if needs_unlabeled_structure:
                    teacher_probs = torch.softmax(teacher_logits, dim=1)
                    teacher_conf, teacher_pseudo = teacher_probs.max(dim=1)
                    teacher_valid = (teacher_conf >= self.proto_conf_threshold) & (teacher_pseudo > 0)
                    if self.dual_space is None:
                        teacher_pseudo = teacher_pseudo.masked_fill(~teacher_valid, 0)
                    teacher_memory_valid = (teacher_conf >= float(self.memory_cfg.get("conf_threshold", self.proto_conf_threshold))) & (teacher_pseudo > 0)
                with torch.no_grad():
                    proto_updates_labeled = self.update_foundation_prototypes(
                        teacher_features_l,
                        gt_masks,
                    )
                    proto_updates = proto_updates + proto_updates_labeled
                    if self.proto_use_unlabeled:
                        proto_updates_unlabeled = self.update_foundation_prototypes(
                            teacher_features_w,
                            teacher_pseudo,
                            confidence=teacher_conf,
                            valid_map=teacher_valid,
                        )
                        proto_updates = proto_updates + proto_updates_unlabeled

                generic_prototype_active = any(
                    coef > 0.0 for coef in (self.proto_region_coef, self.proto_intra_coef, self.proto_sep_coef)
                )
                if generic_prototype_active:
                    proto_region_l, proto_intra_l, proto_sep_loss, proto_regions_labeled = self.region_prototype_losses(
                        student_features_l,
                        gt_masks,
                    )
                if generic_prototype_active and self.proto_use_unlabeled:
                    proto_region_u, proto_intra_u, proto_sep_u, proto_regions_unlabeled = self.region_prototype_losses(
                        student_features_s,
                        teacher_pseudo,
                        confidence=teacher_conf,
                        valid_map=teacher_valid,
                    )
                    proto_region_loss = proto_region_l + self.proto_unlabeled_weight * proto_region_u
                    proto_intra_loss = proto_intra_l + self.proto_unlabeled_weight * proto_intra_u
                    proto_sep_loss = proto_sep_loss + self.proto_unlabeled_weight * proto_sep_u
                elif generic_prototype_active:
                    proto_region_loss = proto_region_l
                    proto_intra_loss = proto_intra_l

                if self.memory_active():
                    with torch.no_grad():
                        memory_updates_labeled = self.update_instance_memory(teacher_features_l, gt_masks)
                        memory_updates = memory_updates_labeled
                        if self.memory_use_unlabeled:
                            memory_updates_unlabeled = self.update_instance_memory(
                                teacher_features_w,
                                teacher_pseudo,
                                valid_map=teacher_memory_valid,
                                confidence=teacher_conf,
                            )
                            memory_updates = memory_updates + memory_updates_unlabeled
                    memory_logits_l, memory_feature_l, memory_regions_l = self.structure_aware_memory_loss(
                        pred_masks_l, student_features_l, gt_masks
                    )
                    memory_logits_loss = memory_logits_l
                    memory_feature_loss = memory_feature_l
                    memory_regions = memory_regions_l
                    memory_regions_labeled = memory_regions_l
                    if self.memory_use_unlabeled:
                        memory_logits_u, memory_feature_u, memory_regions_u = self.structure_aware_memory_loss(
                            student_logits,
                            student_features_s,
                            teacher_pseudo,
                            valid_map=teacher_memory_valid,
                        )
                        memory_logits_loss = memory_logits_l + self.memory_unlabeled_weight * memory_logits_u
                        memory_feature_loss = memory_feature_l + self.memory_unlabeled_weight * memory_feature_u
                        memory_regions = memory_regions_l + memory_regions_u
                        memory_regions_unlabeled = memory_regions_u

            if use_geometry:
                geometry_targets = self.geometry_targets(gt_masks)
                (
                    geometry_boundary_loss,
                    geometry_distance_loss,
                    geometry_separator_head_loss,
                    geometry_boundary_density,
                    geometry_separator_density,
                ) = self.supervised_geometry_losses(geometry_l, gt_masks, targets=geometry_targets)
                # SDF scale is unchanged by the separator extension; retain a
                # diagnostic magnitude without adding it to the optimisation.
                geometry_sdf_magnitude = geometry_targets[1].abs().mean()
                geometry_separator_logit_loss = self.separator_logit_loss(
                    pred_masks_l, geometry_targets[2]
                )
                geometry_hard_loss, geometry_unsup_loss, geometry_hard_fraction = self.hard_boundary_losses(
                    pred_masks_l,
                    geometry_l,
                    gt_masks,
                    geometry_targets[0],
                    teacher_logits=teacher_logits,
                    teacher_geometry=teacher_geometry_w,
                )
                geometry_boundary_consistency = self.boundary_consistency_loss(
                    geometry_s, teacher_logits, teacher_geometry_w
                )

            bsr_loss = pred_masks_l.sum() * 0
            if self.bsr_enabled and use_geometry:
                from network.bsr import boundary_weighted_ce
                bsr_loss = boundary_weighted_ce(pred_masks_l, gt_masks,
                    strength=float(self.bsr_cfg.get("weight_strength", 4.0)),
                    sigma=float(self.bsr_cfg.get("sigma_pixels", 3.0)), ignore_index=self.ignore_index)
            self.log("Loss/train_bsr_weighted_ce", bsr_loss, on_epoch=True, on_step=True, sync_dist=True)
            proto_total = (
                self.bsr_ce_coef * bsr_loss
                + self.proto_region_coef * proto_region_loss
                + self.proto_intra_coef * proto_intra_loss
                + self.proto_sep_coef * proto_sep_loss
                + self.memory_logits_coef * memory_logits_loss
                + self.memory_feature_coef * memory_feature_loss
                + self.geometry_boundary_coef * geometry_boundary_loss
                + self.geometry_distance_coef * geometry_distance_loss
                + self.geometry_hard_ce_coef * geometry_hard_loss
                + self.geometry_unsup_coef * geometry_unsup_loss
                + self.geometry_boundary_consistency_coef * geometry_boundary_consistency
                + self.geometry_separator_head_coef * geometry_separator_head_loss
                + self.geometry_separator_logit_coef * geometry_separator_logit_loss
            )
            total_loss = losses["loss"] + coef * mt_loss + proto_total

            self.log_losses(losses, "train")
            self.log("Loss/train_mt", mt_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_mt_coef", torch.tensor(coef, device=self.device), on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_proto_region", proto_region_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_proto_intra", proto_intra_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_proto_sep", proto_sep_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_proto_total", proto_total, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_memory_logits", memory_logits_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_memory_feature", memory_feature_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_geometry_boundary", geometry_boundary_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_geometry_distance", geometry_distance_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_geometry_hard", geometry_hard_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_geometry_unsup", geometry_unsup_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_boundary_consistency", geometry_boundary_consistency, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_separator_head", geometry_separator_head_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_separator_logits", geometry_separator_logit_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Stats/train_separator_density", geometry_separator_density, on_epoch=True, on_step=True, sync_dist=True)
            if self.dual_space is None:
                self.log("Prototype/updates", proto_updates, on_epoch=True, on_step=True, sync_dist=True)
                self.log("Prototype/updates_labeled", proto_updates_labeled, on_epoch=True, on_step=True, sync_dist=True)
                self.log("Prototype/updates_unlabeled", proto_updates_unlabeled, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Prototype/regions_labeled", proto_regions_labeled, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Prototype/regions_unlabeled", proto_regions_unlabeled, on_epoch=True, on_step=True, sync_dist=True)
            if self.dual_space is None:
                self.log("Memory/updates", memory_updates, on_epoch=True, on_step=True, sync_dist=True)
                self.log("Memory/updates_labeled", memory_updates_labeled, on_epoch=True, on_step=True, sync_dist=True)
                self.log("Memory/updates_unlabeled", memory_updates_unlabeled, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Memory/regions", memory_regions, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Memory/regions_labeled", memory_regions_labeled, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Memory/regions_unlabeled", memory_regions_unlabeled, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Geometry/boundary_density", geometry_boundary_density, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Geometry/sdf_magnitude", geometry_sdf_magnitude, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Geometry/hard_fraction", geometry_hard_fraction, on_epoch=True, on_step=True, sync_dist=True)
            if self.proto_enabled:
                ready = self.dual_space.ready if self.dual_space is not None else self.foundation_proto_ready
                self.log("Prototype/ready_classes", ready[1:].float().sum(), on_epoch=True, on_step=True, sync_dist=True)
                if self.dual_space is None:
                    self.log_prototype_cosines()
                else:
                    for c in range(1, self.num_classes + 1):
                        self.log(f"Memory/occupancy_class_{c}", self.dual_space.memory_ready[c].float().sum(), on_step=True, on_epoch=True)
                    for key, value in self.dual_space.diagnostics.items():
                        self.log(f"DualSpace/{key}", value, on_step=True, on_epoch=True)
            self.log("Loss/train_total", total_loss, on_epoch=True, on_step=True, sync_dist=True)

            mask_cls_pred, gt_masks = self.predict_mask(pred_masks_l, gt_masks, ignored_masks=ignored_masks)
            self.train_metrics.update(mask_cls_pred, gt_masks)
            return total_loss

        images, gt_masks = batch
        gt_masks, ignored_masks = self.process_masks(gt_masks)

        pred_masks, iou_predictions = self(images)
        losses = self.calc_loss(pred_masks, gt_masks, iou_predictions, ignored_masks=ignored_masks, images=images)

        self.log_losses(losses, "train")

        mask_cls_pred, gt_masks = self.predict_mask(pred_masks, gt_masks, ignored_masks=ignored_masks)
        self.train_metrics.update(mask_cls_pred, gt_masks)
        return losses["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.semi_enabled or self.teacher_model is None:
            return
        with torch.no_grad():
            if self.dual_space is not None:
                proto_count, memory_count = self.dual_space.flush(self)
                self.log("DualSpace/prototype_classes_committed", proto_count, on_step=True, on_epoch=True)
                self.log("Memory/committed_updates", memory_count, on_step=True, on_epoch=True)
                for key, value in self.dual_space.diagnostics.items():
                    if key.startswith('memory_') or key == 'upstream_gradient_norm':
                        self.log(f"DualSpace/{key}", value, on_step=True, on_epoch=True)
            # With gradient accumulation, EMA follows optimizer updates only.
            if self._trainer is not None:
                step = int(self.global_step)
                if step == getattr(self, '_last_ema_step', 0):
                    return
                self._last_ema_step = step
            students = dict(self.model.named_parameters())
            teachers = dict(self.teacher_model.named_parameters())
            if students.keys() != teachers.keys():
                raise RuntimeError('Teacher/student parameter names differ')
            for name, teacher_param in teachers.items():
                if students[name].requires_grad:
                    teacher_param.mul_(self.ema_decay).add_(students[name], alpha=1.0 - self.ema_decay)
            student_buffers = dict(self.model.named_buffers())
            for name, teacher_buffer in self.teacher_model.named_buffers():
                teacher_buffer.copy_(student_buffers[name])

    def on_train_epoch_end(self):
        # Log only at epoch end.  Logging a running MetricCollection from each
        # batch produced ambiguous duplicate reductions in TensorBoard.
        self.log_metric_collection(self.train_metrics)
        self.train_metrics.reset()

    def log_metric_collection(self, metric_collection):
        """Log scalar metrics and expand per-class Dice into scalar tags."""
        metric_values = metric_collection.compute()
        per_class_key = next((key for key in metric_values if key.startswith("Dice_Per_Class/")), None)
        if per_class_key is not None:
            per_class_dice = metric_values.pop(per_class_key)
            split = per_class_key.rsplit("/", 1)[1]
            for class_id, dice in enumerate(per_class_dice):
                self.log(
                    f"Dice_Class_{class_id}/{split}", dice,
                    on_step=False, on_epoch=True, sync_dist=True,
                    add_dataloader_idx=False,
                )
        self.log_dict(metric_values, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

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
                self.log_metric_collection(self.eval_metrics)
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
        self.log_metric_collection(self.test_metrics)
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
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters were configured.")
        optimizer = torch.optim.AdamW(trainable_parameters, lr=self.lr, weight_decay=self.hparams.weight_decay)
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
