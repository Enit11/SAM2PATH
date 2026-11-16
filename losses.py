import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

# To deal with unlabeled region for losses: make prediciton & target all zero
# To deal with unlabeled region for metrics: torchmetrics support ignore_index

class SAMLoss(nn.Module):

    def __init__(
        self,
        focal_cof: float = 20.,
        dice_cof: float = 1.,
        ce_cof: float = 0.,
        iou_cof: float = 1.,
        rc_cof: float = 0.,
        rc_sigma: float = 0.5,
        ignore_index: int | None = None,
        class_weights: list[float] | None = None,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.focal_cof = focal_cof
        self.dice_cof = dice_cof
        self.ce_cof = ce_cof
        self.iou_cof = iou_cof
        self.rc_cof = rc_cof
        self.rc_sigma = rc_sigma
        self.ignore_index = ignore_index
        self.focal_gamma = float(focal_gamma)
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

        self.dice_loss_fn = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=False, softmax=False)

    @torch.no_grad()
    def to_one_hot_label(self, targets, num_classes):
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_one_hot = torch.movedim(targets_one_hot, -1, 1)
        return targets_one_hot

    def region_consistency_loss(self, probs, targets, images=None):
        same_x = (targets[:, :, 1:] == targets[:, :, :-1]) & (targets[:, :, 1:] > 0)
        same_y = (targets[:, 1:, :] == targets[:, :-1, :]) & (targets[:, 1:, :] > 0)

        diff_x = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).mean(dim=1)
        diff_y = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).mean(dim=1)

        weight_x = same_x.float()
        weight_y = same_y.float()
        if images is not None:
            img_diff_x = (images[:, :, :, 1:] - images[:, :, :, :-1]).pow(2).mean(dim=1)
            img_diff_y = (images[:, :, 1:, :] - images[:, :, :-1, :]).pow(2).mean(dim=1)
            sigma2 = max(float(self.rc_sigma) ** 2, 1e-6)
            weight_x = weight_x * torch.exp(-img_diff_x / sigma2)
            weight_y = weight_y * torch.exp(-img_diff_y / sigma2)

        loss_x = (diff_x * weight_x).sum() / weight_x.sum().clamp_min(1.0)
        loss_y = (diff_y * weight_y).sum() / weight_y.sum().clamp_min(1.0)
        return 0.5 * (loss_x + loss_y)

    def class_balanced_focal_loss(self, inputs, targets, valid_mask):
        """Apply focal modulation and the configured class weight at each pixel."""
        log_probs = F.log_softmax(inputs, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = -(1.0 - log_pt.exp()).pow(self.focal_gamma) * log_pt
        if self.class_weights is not None:
            focal = focal * self.class_weights[targets]
        focal = focal * valid_mask.float()
        return focal.sum() / valid_mask.float().sum().clamp_min(1.0)

    def forward(self, inputs, targets, iou_pred, ignored_masks=None, images=None):
        # Fail early with a useful message instead of an asynchronous CUDA
        # device-side assert from one_hot/gather when a mask contains an
        # invalid class id.
        if targets.dtype != torch.long:
            targets = targets.long()
        n_channels = int(inputs.shape[1])
        valid_targets = targets if self.ignore_index is None else targets[targets != self.ignore_index]
        if valid_targets.numel() and (int(valid_targets.min()) < 0 or int(valid_targets.max()) >= n_channels):
            raise ValueError(
                f"GT label out of range: min={int(valid_targets.min())}, "
                f"max={int(valid_targets.max())}, model channels={n_channels}, "
                f"ignore_index={self.ignore_index}. Expected labels "
                f"0..{n_channels - 1}."
            )
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        if ignored_masks is not None:
            valid_mask = ~ignored_masks.squeeze(1).bool()

        safe_targets = targets.masked_fill(~valid_mask, 0)
        targets_one_hot = self.to_one_hot_label(safe_targets, num_classes=inputs.shape[1]).float()

        inputs_softmax = F.softmax(inputs, dim=1)

        valid_channels = valid_mask.unsqueeze(1).float()
        dice = self.dice_loss_fn(inputs_softmax * valid_channels, targets_one_hot * valid_channels)
        focal = self.class_balanced_focal_loss(inputs, safe_targets, valid_mask)


        iou_true = calc_iou(inputs_softmax * valid_channels, targets_one_hot * valid_channels)
        iou = F.mse_loss(iou_pred[:, 1:], iou_true[:, 1:]) # ignore background
        # iou = 0.

        ce_kwargs = {"weight": self.class_weights}
        if self.ignore_index is not None:
            ce_kwargs["ignore_index"] = self.ignore_index
        ce_loss = F.cross_entropy(inputs, targets, **ce_kwargs)
        rc_loss = self.region_consistency_loss(inputs_softmax, safe_targets, images=images)

        total_loss = (
            self.focal_cof * focal
            + self.dice_cof * dice
            + self.ce_cof * ce_loss
            + self.iou_cof * iou
            + self.rc_cof * rc_loss
        )

        all_loss = {
            "loss": total_loss,
            "focal": focal,
            "dice": dice,
            "ce": ce_loss,
            "iou": iou,
            "rc": rc_loss,
        }
        # print(f"sim try get total loss: {all_loss}")
        
        return all_loss


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    # both are B, N_cls, H, W
    # pred_mask = F.softmax(pred_mask, dim=1)
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(2, 3))
    union = torch.sum(pred_mask, dim=(2, 3)) + torch.sum(gt_mask, dim=(2, 3)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(2)
    return batch_iou
