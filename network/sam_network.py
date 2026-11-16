import copy
from functools import reduce
from operator import mul
import math
import os

from attrs import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything.modeling.common import LayerNorm2d


from sam2_train.build_sam import build_sam2
# from kan import KAN
from .kan import KAN


SAM2_CFG_BASE_PLUS = "sam2_hiera_b+"
SAM2_CHECKPOINT_BASE_PLUS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "checkpoints",
    "sam2.1_hiera_base_plus.pt",
)
class SharedBackboneDualBranchAdapter(nn.Module):
    """Shared stem with separate structure and boundary branches."""

    def __init__(self, channels: int = 256, boundary_channels: int = 64, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.shared_stem = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )

        self.structure_local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.structure_medium = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.structure_context = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, bias=False)
        self.structure_fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            LayerNorm2d(channels),
        )
        nn.init.zeros_(self.structure_fuse[0].weight)

        self.boundary_branch = nn.Sequential(
            nn.Conv2d(channels, boundary_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(boundary_channels),
            nn.GELU(),
            nn.Conv2d(boundary_channels, boundary_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(boundary_channels),
            nn.GELU(),
        )

    def forward(self, features):
        shared = self.shared_stem(features)
        structure_multi_scale = torch.cat(
            [
                self.structure_local(shared),
                self.structure_medium(shared),
                self.structure_context(shared),
            ],
            dim=1,
        )
        structure_features = shared + self.residual_scale * self.structure_fuse(structure_multi_scale)
        boundary_features = self.boundary_branch(shared)
        return structure_features, boundary_features


class HighResolutionMaskRefiner(nn.Module):
    """Zero-initialized residual refinement using a SAM2 FPN feature."""

    def __init__(self, num_mask_classes: int, feature_channels: int = 256,
                 hidden_channels: int = 64, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=1, bias=False),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.logit_projection = nn.Sequential(
            nn.Conv2d(num_mask_classes, hidden_channels, kernel_size=1, bias=False),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, num_mask_classes, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, low_res_logits: torch.Tensor, high_res_features: torch.Tensor) -> torch.Tensor:
        target_size = low_res_logits.shape[-2:]
        if high_res_features.shape[-2:] != target_size:
            high_res_features = F.interpolate(
                high_res_features, size=target_size, mode="bilinear", align_corners=False
            )
        features = self.feature_projection(high_res_features)
        logits = self.logit_projection(low_res_logits)
        residual = self.residual_head(torch.cat((features, logits), dim=1))
        return low_res_logits + self.residual_scale * residual


class SphericalGatedFusion(nn.Module):
    """Residual SAM2/UNI fusion on a unit hypersphere.

    The original concat-convolution neck remains the reference fusion path.
    This module learns an *additive* correction from separately projected SAM2
    and UNI features.  Its last projection is zero initialized, so enabling it
    leaves an initialization-time model exactly equivalent to the existing
    concat neck.  This is important when initializing from a trained checkpoint.
    """

    def __init__(
        self,
        sam_channels: int = 256,
        uni_channels: int = 256,
        fusion_channels: int = 256,
        projection_channels: int = 256,
        residual_scale: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.eps = float(eps)
        self.sam_projection = nn.Sequential(
            nn.Conv2d(sam_channels, projection_channels, kernel_size=1, bias=False),
            LayerNorm2d(projection_channels),
        )
        self.uni_projection = nn.Sequential(
            nn.Conv2d(uni_channels, projection_channels, kernel_size=1, bias=False),
            LayerNorm2d(projection_channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(projection_channels * 2, projection_channels, kernel_size=1, bias=False),
            LayerNorm2d(projection_channels),
            nn.GELU(),
            nn.Conv2d(projection_channels, 1, kernel_size=1, bias=True),
        )
        self.residual_projection = nn.Sequential(
            nn.Conv2d(projection_channels, fusion_channels, kernel_size=1, bias=False),
            LayerNorm2d(fusion_channels),
        )
        # Preserve the concat-neck baseline exactly at initialization.
        nn.init.zeros_(self.residual_projection[0].weight)

    def forward(self, sam_features: torch.Tensor, uni_features: torch.Tensor):
        if sam_features.shape[0] != uni_features.shape[0] or sam_features.shape[-2:] != uni_features.shape[-2:]:
            raise ValueError(
                "Spherical fusion requires aligned batch/spatial shapes, got "
                f"SAM={tuple(sam_features.shape)} and UNI={tuple(uni_features.shape)}."
            )
        sam_unit = F.normalize(self.sam_projection(sam_features), p=2, dim=1, eps=self.eps)
        uni_unit = F.normalize(self.uni_projection(uni_features), p=2, dim=1, eps=self.eps)
        gate = torch.sigmoid(self.gate(torch.cat((sam_unit, uni_unit), dim=1)))
        spherical_feature = F.normalize(
            gate * sam_unit + (1.0 - gate) * uni_unit, p=2, dim=1, eps=self.eps
        )
        residual = self.residual_projection(spherical_feature)
        return residual, gate, spherical_feature


class BoundaryFeatureCalibration(nn.Module):
    """Residual calibration driven by boundary features and local uncertainty proxy."""
    def __init__(self, structure_channels=256, boundary_channels=64, residual_scale=0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.boundary_projection = nn.Conv2d(boundary_channels, structure_channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(structure_channels * 2, structure_channels, 1, bias=False),
            LayerNorm2d(structure_channels), nn.GELU(), nn.Conv2d(structure_channels, 1, 1),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(structure_channels, structure_channels, 3, padding=1, bias=False),
            LayerNorm2d(structure_channels), nn.GELU(), nn.Conv2d(structure_channels, structure_channels, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, structure, boundary):
        boundary = self.boundary_projection(boundary)
        gate = torch.sigmoid(self.gate(torch.cat((structure, boundary), dim=1)))
        return structure + self.residual_scale * gate * self.residual(structure), gate


class PromptSAM(nn.Module):

    def __init__(
            self,
            model_type: str = "vit_b",
            checkpoint: str = "",
            prompt_dim: int = 256,
            num_classes: int = 20,
            extra_encoder = None,
            freeze_image_encoder = True,
            freeze_prompt_encoder = True,
            freeze_mask_decoder = False,
            mask_HW = (1024, 1024),
            feature_input = False,
            prompt_decoder = False,
            dense_prompt_decoder=False,
            no_sam=False
    ):
        super().__init__()

        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.mask_HW = mask_HW
        self.feature_input = feature_input

        self.extra_encoder = extra_encoder
        # change prompt
        mask_tokens = nn.Embedding(num_classes + 1, prompt_dim)
        self.model.mask_decoder.mask_tokens = mask_tokens
        self.model.mask_decoder.num_mask_tokens = num_classes + 1

        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList(
            [
                # self.model.mask_decoder.output_hypernetworks_mlps[0].clone()
                copy.deepcopy(self.model.mask_decoder.output_hypernetworks_mlps[0])
                for i in range(self.model.mask_decoder.num_mask_tokens)
            ]
        )

        self.model.mask_decoder.iou_prediction_head.layers[-1] = nn.Linear(prompt_dim,
                                                                           self.model.mask_decoder.num_mask_tokens)

        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        self.dense_prompt_decoder = None
        if dense_prompt_decoder:
            decoder_layer = nn.TransformerDecoderLayer(d_model=prompt_dim, nhead=8)
            self.dense_prompt_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.no_sam = no_sam


    def forward(self, images, return_feature=False):
        H, W = self.mask_HW

        if not self.feature_input:
            if images.shape[-2] != 1024 or images.shape[-1] != 1024:
                images = F.interpolate(images, (1024, 1024), mode="bilinear", align_corners=False)

            if not self.no_sam:
                with torch.no_grad():
                    image_embeddings = self.model.image_encoder(images)

        if self.extra_encoder is not None:
            extra_image_embeddings = self.extra_encoder(images)
            if self.no_sam:
                image_embeddings = extra_image_embeddings
                # print(image_embeddings.shape)
            else:
                image_embeddings = image_embeddings + extra_image_embeddings

        pred_masks = []
        ious = []
        for embedding in image_embeddings: #zip(image_embeddings):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

            if self.dense_prompt_decoder is not None:
                # img embedding dim, H, W ->  HW, dim
                embedding_img = embedding.flatten(1).permute(1, 0)
                # sparse_embeddings: Ncls + 1, dim ->  Ncls, dim
                sparse_embeddings_v = self.model.mask_decoder.mask_tokens.weight.clone()
                # org dense_embeddings shape: 1, 256, 64, 64, now it is 4094, 256
                org_shape = dense_embeddings.shape
                dense_embeddings_gen = self.dense_prompt_decoder(embedding_img, sparse_embeddings_v)
                dense_embeddings_gen = dense_embeddings_gen.permute(1, 0).reshape(*org_shape)
                dense_embeddings = dense_embeddings + dense_embeddings_gen

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(0))
            ious.append(iou_predictions.reshape(-1, 1))

        if return_feature:
            return pred_masks, ious, image_embeddings
        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)

class PromptSAMLateFusion(nn.Module):
    def __init__(
            self,
            model_type: str = "vit_b",
            checkpoint: str = "",
            prompt_dim: int = 256,
            num_classes: int = 20,
            extra_encoder = None,
            freeze_image_encoder = True,
            freeze_prompt_encoder = True,
            freeze_mask_decoder = False,
            mask_HW = (1024, 1024),
            feature_input = False,
            prompt_decoder = False,
            dense_prompt_decoder=False,
            no_sam=False,
            extra_encoder_dim=None,
            geometry_cfg=None,
    ):
        super().__init__()
        if build_sam2 is None:
            raise ImportError("SAM2 dependencies are unavailable, so PromptSAMLateFusion cannot be built.")
        sam2_model = build_sam2(
            config_file=SAM2_CFG_BASE_PLUS,
            ckpt_path=SAM2_CHECKPOINT_BASE_PLUS,
            mode="train",
        )
        image_encoder2 = sam2_model.image_encoder
        del sam2_model
        self.model = sam_model_registry[model_type](checkpoint=checkpoint) if checkpoint else sam_model_registry[model_type]()
        self.mask_HW = mask_HW
        self.feature_input = feature_input
        
        self.extra_encoder = extra_encoder
        self.extra_encoder_dim = extra_encoder_dim
        # change sam2 encoder
        self.model.image_encoder = image_encoder2
        # change prompt
        mask_tokens = nn.Embedding(num_classes + 1, prompt_dim)  # -->  (7, 256)
        self.model.mask_decoder.mask_tokens = mask_tokens
        self.model.mask_decoder.num_mask_tokens = num_classes + 1
        # sim added todo: 这里将MLPs 修改为KANs
        # mlp : 256 --> 32  kan  
        ########## 使用 KAN ##################
        self.kan1 = KAN([256, 32])  # mlp : 256 --> 32  kan  256 --> 32 simzhangbest modified
        self.kan2 = KAN([256, num_classes + 1])
        # self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         # self.model.mask_decoder.output_hypernetworks_mlps[0].clone()
        #         copy.deepcopy(self.model.mask_decoder.output_hypernetworks_mlps[0])
        #         for i in range(self.model.mask_decoder.num_mask_tokens)
        #     ]
        # )

        # self.model.mask_decoder.iou_prediction_head.layers[-1] = nn.Linear(prompt_dim,
        #                                                                    self.model.mask_decoder.num_mask_tokens)  # 输出到classes

             
        
        new_hypernetworks_mlps = nn.ModuleList(
            [copy.deepcopy(self.kan1) for _ in range(self.model.mask_decoder.num_mask_tokens)]
        )

        # 替换原有的ModuleList
        self.model.mask_decoder.output_hypernetworks_mlps = new_hypernetworks_mlps
        
        self.model.mask_decoder.iou_prediction_head = self.kan2
                
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        self.fusion_neck = None
        if self.extra_encoder is not None:
            if self.extra_encoder_dim is None:
                raise ValueError("extra_encoder_dim is required when extra_encoder is enabled.")
            self.fusion_neck = nn.Sequential(
                nn.Conv2d(
                    256 + self.extra_encoder_dim,
                    256,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(256),
                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(256),
            )
        # The SAM2 encoder stays frozen; this adapter is the trainable feature space used by
        # both the mask decoder and structure/prototype losses.
        self.structure_feature_adapter = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

        self.dense_prompt_decoder = None
        if dense_prompt_decoder:
            decoder_layer = nn.TransformerDecoderLayer(d_model=prompt_dim, nhead=8)
            self.dense_prompt_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

    def forward(self, images, return_feature=False):
        H, W = self.mask_HW

        if not self.feature_input:
            if images.shape[-2] != 1024 or images.shape[-1] != 1024:
                images = F.interpolate(images, (1024, 1024), mode="bilinear", align_corners=False)  # torch.Size([1, 3, 1024, 1024])

            with torch.no_grad():
                image_embeddings = self.model.image_encoder(images)["vision_features"]
                
                

        if self.extra_encoder is not None:
            ex_embed = self.extra_encoder(images)
            print("[DEBUG] image_embeddings", image_embeddings.shape)
            print("[DEBUG] ex_embed_before_reshape", ex_embed.shape)
            ex_embed = ex_embed.reshape(ex_embed.shape[0], 64, 64, -1).permute(0, 3, 1, 2)
            print("[DEBUG] ex_embed_after_reshape", ex_embed.shape)
            image_embeddings = torch.cat((image_embeddings, ex_embed), dim=1)
            print("[DEBUG] fused_before_neck", image_embeddings.shape)
            image_embeddings = self.fusion_neck(image_embeddings)

        # This creates a learnable segmentation feature even when the SAM2 encoder is frozen.
        image_embeddings = self.structure_feature_adapter(image_embeddings)

        pred_masks = []
        ious = []
        for embedding in image_embeddings: #zip(image_embeddings):  根据batch 来放进去，一次处理一张图片
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

            if self.dense_prompt_decoder is not None:
                # img embedding dim, H, W ->  HW, dim
                embedding_img = embedding.flatten(1).permute(1, 0)
                # sparse_embeddings: Ncls + 1, dim ->  Ncls, dim
                sparse_embeddings_v = self.model.mask_decoder.mask_tokens.weight.clone()
                #org dense_embeddings shape: 1, 256, 64, 64, now it is 4094, 256
                org_shape = dense_embeddings.shape
                dense_embeddings_gen = self.dense_prompt_decoder(embedding_img, sparse_embeddings_v)
                dense_embeddings_gen = dense_embeddings_gen.permute(1, 0).reshape(*org_shape)
                dense_embeddings = dense_embeddings + dense_embeddings_gen

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),   # [1, 256, 64, 64] --> [256, 64, 64]
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )  # low_res_masks: [1, 3, 256, 256]  --> iou prediction   iou_predictions: [1, 3] classes prediction

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )  # masks: [1, 3, 1536, 1536]
            pred_masks.append(masks.squeeze(0))
            ious.append(iou_predictions.reshape(-1, 1))

        if return_feature:
            return pred_masks, ious, image_embeddings
        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)
    
    
    

class PromptSAMLateFusion_NOKAN(nn.Module):

    def __init__(
            self,
            model_type: str = "vit_b",
            checkpoint: str = "",
            prompt_dim: int = 256,
            num_classes: int = 20,
            extra_encoder=None,
            freeze_image_encoder=True,
            freeze_prompt_encoder=True,
            freeze_mask_decoder=False,
            mask_HW=(1024, 1024),
            feature_input=False,
            prompt_decoder=False,
            dense_prompt_decoder=False,
            no_sam=False,
            extra_encoder_dim=None,
            geometry_cfg=None,
            high_res_refinement_cfg=None,
            spherical_fusion_cfg=None,
    ):
        super().__init__()

        # ============================================================
        # Basic configuration
        # ============================================================
        self.mask_HW = mask_HW
        self.feature_input = feature_input
        self.extra_encoder = extra_encoder
        self.extra_encoder_dim = extra_encoder_dim
        self.geometry_cfg = geometry_cfg or {}
        self.high_res_refinement_cfg = high_res_refinement_cfg or {}
        self.spherical_fusion_cfg = spherical_fusion_cfg or {}

        # Debug: only print fusion information once
        self._fusion_debug_printed = False

        # ============================================================
        # Build SAM2.1 Hiera Base+ image encoder
        # ============================================================
        if build_sam2 is None:
            raise ImportError(
                "SAM2 dependencies are unavailable, "
                "so PromptSAMLateFusion_NOKAN cannot be built."
            )

        sam2_model = build_sam2(
            config_file=SAM2_CFG_BASE_PLUS,
            ckpt_path=SAM2_CHECKPOINT_BASE_PLUS,
            mode="train",
        )

        image_encoder2 = sam2_model.image_encoder

        # SAM2 model object no longer needed
        del sam2_model

        # ============================================================
        # Build original SAM container
        # We only replace its image encoder with SAM2 encoder
        # ============================================================
        if checkpoint:
            self.model = sam_model_registry[model_type](
                checkpoint=checkpoint
            )
        else:
            self.model = sam_model_registry[model_type]()

        # Replace original SAM image encoder with SAM2 encoder
        self.model.image_encoder = image_encoder2

        # ============================================================
        # Basic settings
        # ============================================================
        self.no_sam = no_sam

        # ============================================================
        # Change mask tokens
        # Background + num_classes foreground classes
        # ============================================================
        mask_tokens = nn.Embedding(
            num_classes + 1,
            prompt_dim
        )

        self.model.mask_decoder.mask_tokens = mask_tokens
        self.model.mask_decoder.num_mask_tokens = num_classes + 1

        # ============================================================
        # NOKAN:
        # Keep original MLP hypernetworks
        # ============================================================
        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList(
            [
                copy.deepcopy(
                    self.model.mask_decoder
                    .output_hypernetworks_mlps[0]
                )
                for _ in range(
                    self.model.mask_decoder.num_mask_tokens
                )
            ]
        )

        # ============================================================
        # IoU prediction head
        # Change final output dimension to num_classes + 1
        # ============================================================
        self.model.mask_decoder.iou_prediction_head.layers[-1] = nn.Linear(
            prompt_dim,
            self.model.mask_decoder.num_mask_tokens
        )

        # ============================================================
        # Freeze / train settings
        # ============================================================
        if freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False

        if freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False

        if freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

        # ============================================================
        # SAM2 + UNI2-h Fusion Neck
        #
        # SAM2:
        #   [B, 256, 64, 64]
        #
        # UNI2-h:
        #   [B, 1536, 16, 16]
        #
        # UNI2-h is interpolated to:
        #   [B, 1536, 64, 64]
        #
        # concat:
        #   [B, 1792, 64, 64]
        #
        # fusion neck:
        #   [B, 256, 64, 64]
        # ============================================================
        self.fusion_neck = None
        self.spherical_fusion = None

        if self.extra_encoder is not None:

            if self.extra_encoder_dim is None:
                raise ValueError(
                    "extra_encoder_dim is required "
                    "when extra_encoder is enabled."
                )

            self.fusion_neck = nn.Sequential(
                nn.Conv2d(
                    256 + self.extra_encoder_dim,
                    256,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(256),

                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(256),
            )
            if bool(self.spherical_fusion_cfg.get("enabled", False)):
                if self.spherical_fusion_cfg.get("mode", "spherical") == "bidirectional":
                    from network.bidirectional_fusion import BidirectionalGatedFusion
                    self.spherical_fusion = BidirectionalGatedFusion(
                        sam_channels=256, uni_channels=self.extra_encoder_dim,
                        window_size=int(self.spherical_fusion_cfg.get("window_size", 8)),
                        heads=int(self.spherical_fusion_cfg.get("heads", 4)),
                        residual_scale=float(self.spherical_fusion_cfg.get("residual_scale", 0.1)),
                    )
                else:
                    self.spherical_fusion = SphericalGatedFusion(
                    sam_channels=256,
                    uni_channels=self.extra_encoder_dim,
                    fusion_channels=256,
                    projection_channels=int(self.spherical_fusion_cfg.get("projection_channels", 256)),
                    residual_scale=float(self.spherical_fusion_cfg.get("residual_scale", 0.1)),
                    eps=float(self.spherical_fusion_cfg.get("eps", 1e-6)),
                )

        # ============================================================
        # Geometry / Structure Adapter
        # ============================================================
        self.geometry_enabled = bool(
            self.geometry_cfg.get("enabled", False)
        )

        if (
            self.geometry_enabled
            and bool(
                self.geometry_cfg.get(
                    "multi_scale_adapter",
                    True
                )
            )
        ):

            self.structure_feature_adapter = (
                SharedBackboneDualBranchAdapter(
                    channels=256,
                    boundary_channels=int(
                        self.geometry_cfg.get(
                            "head_channels",
                            64
                        )
                    ),
                    residual_scale=float(
                        self.geometry_cfg.get(
                            "residual_scale",
                            0.1
                        )
                    ),
                )
            )

        else:

            self.structure_feature_adapter = nn.Sequential(
                nn.Conv2d(
                    256,
                    256,
                    kernel_size=1,
                    bias=False
                ),
                LayerNorm2d(256),
                nn.GELU(),

                nn.Conv2d(
                    256,
                    256,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                LayerNorm2d(256),
            )

        self.boundary_feature_calibration = None
        if self.geometry_enabled and bool(self.geometry_cfg.get("feature_calibration_enabled", False)):
            self.boundary_feature_calibration = BoundaryFeatureCalibration(
                structure_channels=256,
                boundary_channels=int(self.geometry_cfg.get("head_channels", 64)),
                residual_scale=float(self.geometry_cfg.get("feature_calibration_scale", 0.1)),
            )
            if bool(self.geometry_cfg.get("bsr_enabled", False)):
                from network.bsr import BoundaryStructureRefinement
                self.boundary_feature_calibration = BoundaryStructureRefinement(
                    channels=256, boundary_channels=int(self.geometry_cfg.get("head_channels", 64)),
                    gamma_init=float(self.geometry_cfg.get("bsr_gamma_init", 0.0)),
                )

        # ============================================================
        # Geometry auxiliary heads
        # ============================================================
        if self.geometry_enabled:

            hidden_channels = int(
                self.geometry_cfg.get(
                    "head_channels",
                    64
                )
            )

            # Boundary prediction
            self.boundary_head = nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(hidden_channels),
                nn.GELU(),

                nn.Conv2d(
                    hidden_channels,
                    1,
                    kernel_size=1
                ),
            )

            # Distance prediction
            self.distance_head = nn.Sequential(
                nn.Conv2d(
                    256,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(hidden_channels),
                nn.GELU(),

                # One signed-distance channel per foreground class:
                # channel 0 -> class 1 (glomerulus)
                # channel 1 -> class 2 (vessel)
                # channel 2 -> class 3 (tubule)
                nn.Conv2d(
                    hidden_channels,
                    num_classes,
                    kernel_size=1
                ),
            )

            # Dedicated supervision for GT background gaps between distinct
            # instances of the same semantic class.
            self.separator_head = None
            if bool(self.geometry_cfg.get("separator_enabled", False)):
                self.separator_head = nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                    LayerNorm2d(hidden_channels),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, 1, kernel_size=1),
                )

        # ============================================================
        # Dense prompt decoder
        # ============================================================
        self.high_res_refiner = None
        if bool(self.high_res_refinement_cfg.get("enabled", False)):
            self.high_res_refiner = HighResolutionMaskRefiner(
                num_mask_classes=num_classes + 1,
                feature_channels=256,
                hidden_channels=int(self.high_res_refinement_cfg.get("hidden_channels", 64)),
                residual_scale=float(self.high_res_refinement_cfg.get("residual_scale", 0.1)),
            )

        self.dense_prompt_decoder = None

        if dense_prompt_decoder:

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=prompt_dim,
                nhead=8
            )

            self.dense_prompt_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=1
            )

    # =================================================================
    # Forward
    # =================================================================
    def forward(
            self,
            images,
            return_feature=False,
            return_aux=False
    ):

        H, W = self.mask_HW

        # ============================================================
        # 1. Resize input to SAM2 expected resolution
        # ============================================================
        if not self.feature_input:

            if (
                images.shape[-2] != 1024
                or images.shape[-1] != 1024
            ):
                images = F.interpolate(
                    images,
                    (1024, 1024),
                    mode="bilinear",
                    align_corners=False,
                )

            # ========================================================
            # 2. Frozen SAM2 image encoder
            # ========================================================
            with torch.no_grad():
                backbone_output = self.model.image_encoder(images)
                image_embeddings = backbone_output["vision_features"]
                # Keep SAM2's highest-resolution FPN output for the optional
                # refiner. The SAM decoder itself still consumes 64x64 tokens.
                high_res_features = max(
                    backbone_output["backbone_fpn"],
                    key=lambda feature: feature.shape[-2] * feature.shape[-1],
                )
        elif self.high_res_refiner is not None:
            raise ValueError(
                "high_res_refinement requires image input and SAM2 backbone_fpn features."
            )

        # ============================================================
        # 3. SAM2 + UNI2-h fusion
        # ============================================================
        if self.extra_encoder is not None:

            # UNI2-h
            ex_embed = self.extra_encoder(images)

            # --------------------------------------------------------
            # UNI2-h spatial alignment
            #
            # [B,1536,16,16]
            #       ↓
            # [B,1536,64,64]
            # --------------------------------------------------------
            if ex_embed.shape[-2:] != image_embeddings.shape[-2:]:

                ex_embed = F.interpolate(
                    ex_embed,
                    size=image_embeddings.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            # --------------------------------------------------------
            # Concatenate SAM2 + UNI2
            #
            # SAM2 : [B,256,64,64]
            # UNI2 : [B,1536,64,64]
            #
            # -> [B,1792,64,64]
            # --------------------------------------------------------
            sam_embeddings = image_embeddings
            image_embeddings = torch.cat((sam_embeddings, ex_embed), dim=1)

            # --------------------------------------------------------
            # One-time debug
            # --------------------------------------------------------
            if not self._fusion_debug_printed:

                print("=" * 80)
                print("SAM2 + UNI2 FUSION")
                print(
                    "SAM2 feature :",
                    image_embeddings[:, :256].shape
                )
                print(
                    "UNI2 feature :",
                    image_embeddings[:, 256:].shape
                )
                print(
                    "Fused feature:",
                    image_embeddings.shape
                )

            # --------------------------------------------------------
            # Fusion Neck
            #
            # [B,1792,64,64]
            #        ↓
            # [B,256,64,64]
            # --------------------------------------------------------
            image_embeddings = self.fusion_neck(
                image_embeddings
            )

            # The spherical stream is an explicitly normalized semantic
            # correction, not a replacement for the validated concat neck.
            # Its final projection starts at zero, hence old checkpoints have
            # identical logits before this new stream learns an update.
            if self.spherical_fusion is not None:
                spherical_residual, fusion_gate, _ = self.spherical_fusion(
                    sam_embeddings, ex_embed
                )
                image_embeddings = image_embeddings + self.spherical_fusion.residual_scale * spherical_residual
                if not self._fusion_debug_printed:
                    print("Spherical fusion gate mean:", float(fusion_gate.detach().mean()))

            # --------------------------------------------------------
            # Finish one-time debug
            # --------------------------------------------------------
            if not self._fusion_debug_printed:

                print(
                    "After fusion :",
                    image_embeddings.shape
                )
                print("=" * 80)

                self._fusion_debug_printed = True

        # ============================================================
        # 4. Structure feature adaptation
        #
        # IMPORTANT:
        # This is the feature used by:
        #   - SAM mask decoder
        #   - Prototype loss
        #   - SIPP
        #   - ICP
        #
        # Therefore it remains trainable even though SAM2 is frozen.
        # ============================================================
        structure_features = image_embeddings
        boundary_features = None
        calibration_gate = None

        if self.geometry_enabled:

            adapted = self.structure_feature_adapter(
                image_embeddings
            )

            if isinstance(adapted, tuple):

                structure_features, boundary_features = adapted

            else:

                structure_features = adapted

            if self.boundary_feature_calibration is not None and (boundary_features is not None or self.geometry_cfg.get("bsr_enabled", False)):
                structure_features, calibration_gate = self.boundary_feature_calibration(
                    structure_features, boundary_features
                )

        else:

            structure_features = (
                self.structure_feature_adapter(
                    image_embeddings
                )
            )

        # ============================================================
        # 5. Auxiliary geometry outputs
        # ============================================================
        auxiliary_outputs = {}

        if self.geometry_enabled:

            if boundary_features is None:
                boundary_features = structure_features

            auxiliary_outputs = {

                "boundary_logits": F.interpolate(
                    self.boundary_head(
                        boundary_features
                    ),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ),

                "distance_maps": F.interpolate(
                    self.distance_head(
                        structure_features
                    ),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ),
            }
            if calibration_gate is not None:
                auxiliary_outputs["calibration_gate"] = F.interpolate(
                    calibration_gate, size=(H, W), mode="bilinear", align_corners=False
                )
            if self.separator_head is not None:
                auxiliary_outputs["separator_logits"] = F.interpolate(
                    self.separator_head(boundary_features),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )

        # ============================================================
        # 6. SAM prompt + mask decoder
        # ============================================================
        pred_masks = []
        ious = []

        # Process each image in batch
        for batch_index, embedding in enumerate(structure_features):

            sparse_embeddings, dense_embeddings = (
                self.model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
            )

            # ========================================================
            # Optional dense prompt decoder
            # ========================================================
            if self.dense_prompt_decoder is not None:

                # [256,H,W]
                # ->
                # [HW,256]
                embedding_img = (
                    embedding
                    .flatten(1)
                    .permute(1, 0)
                )

                sparse_embeddings_v = (
                    self.model
                    .mask_decoder
                    .mask_tokens
                    .weight
                    .clone()
                )

                org_shape = dense_embeddings.shape

                dense_embeddings_gen = (
                    self.dense_prompt_decoder(
                        embedding_img,
                        sparse_embeddings_v
                    )
                )

                dense_embeddings_gen = (
                    dense_embeddings_gen
                    .permute(1, 0)
                    .reshape(*org_shape)
                )

                dense_embeddings = (
                    dense_embeddings
                    + dense_embeddings_gen
                )

            # ========================================================
            # SAM mask decoder
            # ========================================================
            low_res_masks, iou_predictions = (
                self.model.mask_decoder(
                    image_embeddings=embedding.unsqueeze(0),

                    image_pe=(
                        self.model
                        .prompt_encoder
                        .get_dense_pe()
                    ),

                    sparse_prompt_embeddings=sparse_embeddings,

                    dense_prompt_embeddings=dense_embeddings,

                    multimask_output=True,

                    # Semantic segmentation uses one logit channel per class:
                    # token 0 is background and tokens 1..N are foreground.
                    # The SAM default for multimask mode intentionally drops
                    # token 0, which would incorrectly leave only 3 channels.
                    return_all_masks=True,
                )
            )

            if self.high_res_refiner is not None:
                low_res_masks = self.high_res_refiner(
                    low_res_masks,
                    high_res_features[batch_index : batch_index + 1],
                )

            # ========================================================
            # Upsample mask
            # ========================================================
            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )

            pred_masks.append(
                masks.squeeze(0)
            )

            ious.append(
                iou_predictions.reshape(-1, 1)
            )

        # ============================================================
        # 7. Return
        # ============================================================
        if return_feature and return_aux:

            return (
                pred_masks,
                ious,
                structure_features,
                auxiliary_outputs,
            )

        if return_feature:

            return (
                pred_masks,
                ious,
                structure_features,
            )

        if return_aux:

            return (
                pred_masks,
                ious,
                auxiliary_outputs,
            )

        return pred_masks, ious

    # =================================================================
    # Predictor
    # =================================================================
    def get_predictor(self):

        return SamPredictor(self.model)
