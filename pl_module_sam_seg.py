from functools import partial
import copy
import math

import torch
from pytorch_lightning import LightningModule
from torch import nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
import time

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

        self.time_and_cnt = [0., 0]
        self.teacher_model = None
        if self.semi_enabled:
            self.teacher_model = copy.deepcopy(self.model)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def forward(self, images):
        # use forward for inference/predictions
        pred_masks, iou_predictions = self.model(images)

        # pred_masks and iou_predictions are lists  将list 变成 torch 张量
        pred_masks = torch.stack(pred_masks, dim=0)
        iou_predictions = torch.stack(iou_predictions, dim=0)

        return pred_masks, iou_predictions

    @torch.no_grad()
    def teacher_forward(self, images):
        pred_masks, iou_predictions = self.teacher_model(images)
        pred_masks = torch.stack(pred_masks, dim=0)
        iou_predictions = torch.stack(iou_predictions, dim=0)
        return pred_masks, iou_predictions

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

    def training_step(self, batch, batch_idx):
        if self.semi_enabled and isinstance(batch, dict):
            (images_l, gt_masks) = batch["labeled"]
            images_w, images_s = batch["unlabeled"]
            gt_masks, ignored_masks = self.process_masks(gt_masks)

            pred_masks_l, iou_predictions_l = self(images_l)
            losses = self.calc_loss(pred_masks_l, gt_masks, iou_predictions_l, ignored_masks=ignored_masks, images=images_l)

            with torch.no_grad():
                teacher_logits, _ = self.teacher_forward(images_w)
            student_logits, _ = self(images_s)
            mt_loss = self.consistency_loss(student_logits, teacher_logits)
            coef = self.mt_coef()
            total_loss = losses["loss"] + coef * mt_loss

            self.log_losses(losses, "train")
            self.log("Loss/train_mt", mt_loss, on_epoch=True, on_step=True, sync_dist=True)
            self.log("Loss/train_mt_coef", torch.tensor(coef, device=self.device), on_epoch=True, on_step=True, sync_dist=True)
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
