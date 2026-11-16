from copy import deepcopy
from functools import partial
import math
from segment_anything.modeling.common import LayerNorm2d

import torch
from torch import nn
import torch.nn.functional as F


class EncoderWrapper(nn.Module):
    def __init__(self, model, ft_dim, out_dim, neck=True, re_norm = False, mean=None, std=None,
                 trainable_neck=False):
        super().__init__()
        self.model = model
        # The foundation Transformer is inference-only.  Its optional neck is
        # configured independently below and may remain trainable.
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.input_dim = ft_dim
        self.output_dim = out_dim if neck else ft_dim
        self.prefix_tokens = int(getattr(model, "num_prefix_tokens", 0) or 0)
        if neck:
            self.neck = nn.Sequential(
                nn.Conv2d(
                    ft_dim,
                    out_dim,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_dim),
                nn.Conv2d(
                    out_dim,
                    out_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_dim),
            )
            for parameter in self.neck.parameters():
                parameter.requires_grad = bool(trainable_neck)
        else:
            self.neck = None

        self.re_norm = re_norm

        self.register_buffer("in_mean", torch.Tensor((0.485, 0.456, 0.406)).view(-1, 1, 1), False)
        self.register_buffer("in_std", torch.Tensor((0.229, 0.224, 0.225)).view(-1, 1, 1), False)
        self.register_buffer("mean", torch.Tensor(mean).view(-1, 1, 1), False)
        self.register_buffer("std", torch.Tensor(std).view(-1, 1, 1), False)

    def forward(self, x, no_grad=True): #  x [B, 3, H, W]
        if self.re_norm:
            x = x * self.in_std + self.in_mean
            x = (x - self.mean) / self.std
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "img_size"):
            target_size = self.model.patch_embed.img_size
            if isinstance(target_size, int):
                target_size = (target_size, target_size)
            if x.shape[-2:] != tuple(target_size):
                x = F.interpolate(x, size=tuple(target_size), mode="bilinear", align_corners=False)
        if no_grad:
            # Freeze only the foundation Transformer.  The optional neck is
            # deliberately evaluated outside this context when trainable so
            # its gradients reach the optimizer.
            with torch.no_grad():
                try:
                    x = self.model(x, dense=True)
                except TypeError:
                    x = self.model.forward_features(x)
        else:
            try:
                x = self.model(x, dense=True)
            except TypeError:
                x = self.model.forward_features(x)
        if x.ndim == 3:
            if self.prefix_tokens > 0 and x.shape[1] > self.prefix_tokens:
                x = x[:, self.prefix_tokens :, :]
            spatial_tokens = x.shape[1]
            grid_size = int(math.isqrt(spatial_tokens))
            if grid_size * grid_size != spatial_tokens:
                raise ValueError(
                    f"Cannot reshape token sequence with length {spatial_tokens} into a square grid."
                )
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], grid_size, grid_size)
        if self.neck is not None:
            if x.shape[-1] != 64 or x.shape[-2] != 64:
                x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
            x = self.neck(x)
        return x


def get_hipt(pretrained=None, neck=True):
    # from .hipt.vision_transformer import vit_small
    # from .hipt.hipt_prompt import load_ssl_weights
    
    # debug use
    from hipt.vision_transformer import vit_small
    from hipt.hipt_prompt import load_ssl_weights

    model = vit_small(patch_size=16)
    model = load_ssl_weights(model, pretrained)

    model = EncoderWrapper(model, 384, 256, neck=neck,
                           re_norm=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    return model


import os
import timm
os.environ['UNI_CKPT_PATH'] = "/mnt/zm/code/CLAM/checkpoints/uni/pytorch_model.bin"

def has_uni():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH

def get_uni(pretrained=None, neck=True, trainable_neck=False):
    from .hipt.vision_transformer import vit_large
    from .hipt.hipt_prompt import load_uni_weights
    model = vit_large()
    model = load_uni_weights(model, pretrained)
    
    model = EncoderWrapper(model, 1024, 256, neck=neck,
                           re_norm=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                           trainable_neck=trainable_neck)
    return model


def get_uni2_h(pretrained=None, neck=True, trainable_neck=False):
    import timm

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "embed_dim": 1536,
        "depth": 24,
        "num_heads": 24,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "global_pool": "token",
        "no_embed_class": True,
        "init_values": 1e-5,
        "qkv_bias": True,
        "act_layer": torch.nn.SiLU,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "class_token": True,
    }

    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **timm_kwargs)

    if pretrained is None:
        pretrained = os.environ.get("UNI2_H_CKPT", None)
    if pretrained is not None:
        if os.path.isdir(pretrained):
            pretrained = os.path.join(pretrained, "pytorch_model.bin")
        state_dict = torch.load(pretrained, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] UNI2-h missing keys: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        if unexpected:
            print(f"[WARN] UNI2-h unexpected keys: {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")

    model = EncoderWrapper(
        model,
        1536,
        256,
        neck=neck,
        re_norm=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        trainable_neck=trainable_neck,
    )
    return model

# sim added for model encoder test
from torchsummary import summary
if __name__ == '__main__':
    # x should be B, 4096, dim
    # pretrained = "/mnt/zmy/code/sam-path/pretrained/vit256_small_dino.pth"
    # encoder = get_hipt(pretrained, neck=True)
    # # x [6, 3, 1024, 1024]
    # input_size = (6, 3, 1024, 1024)
    
    # input_feature = torch.randn(input_size)
    # out = encoder(input_feature)
    # print(out.shape)  # [6, 4096, 384] neck=False   # [6, 256, 64, 64] neck=True
    


    # test uni
    _, pretrained = has_uni()
    encoder = get_uni(pretrained, neck=True)
    input_size = (6, 3, 1024, 1024)
    input_feature = torch.randn(input_size)
    out = encoder(input_feature)
    print(out.shape) # [6, 4096, 1024] neck=False  # 
