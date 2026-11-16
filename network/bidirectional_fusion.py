"""Aligned bidirectional window attention with channel/spatial source selection."""
import torch
from torch import nn
import torch.nn.functional as F


class BidirectionalGatedFusion(nn.Module):
    def __init__(self, sam_channels=256, uni_channels=256, channels=256,
                 window_size=8, heads=4, residual_scale=0.1):
        super().__init__()
        self.window_size = window_size
        self.residual_scale = residual_scale
        self.sam_projection = nn.Conv2d(sam_channels, channels, 1, bias=False)
        self.uni_projection = nn.Conv2d(uni_channels, channels, 1, bias=False)
        self.sam_norm = nn.LayerNorm(channels)
        self.uni_norm = nn.LayerNorm(channels)
        self.sam_attention = nn.MultiheadAttention(channels, heads, batch_first=True, dropout=0.)
        self.uni_attention = nn.MultiheadAttention(channels, heads, batch_first=True, dropout=0.)
        self.channel_gate = nn.Sequential(nn.Conv2d(2*channels, channels//4, 1), nn.GELU(), nn.Conv2d(channels//4, channels, 1))
        self.spatial_gate = nn.Sequential(nn.Conv2d(2*channels, channels//4, 1), nn.GELU(), nn.Conv2d(channels//4, 1, 1))
        self.residual_projection = nn.Conv2d(channels, channels, 1)
        # Zero output preserves the concat neck at warm start. Attention remains
        # nonzero-initialized so the residual projection receives a useful input.
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)
        for gate in (self.channel_gate, self.spatial_gate):
            nn.init.zeros_(gate[-1].weight)
            nn.init.zeros_(gate[-1].bias)

    def windows(self, x):
        b,c,h,w = x.shape
        s = self.window_size
        return x.reshape(b,c,h//s,s,w//s,s).permute(0,2,4,3,5,1).reshape(-1,s*s,c)

    def restore(self, x, shape):
        b,c,h,w = shape
        s = self.window_size
        return x.reshape(b,h//s,w//s,s,s,c).permute(0,5,1,3,2,4).reshape(b,c,h,w)

    def forward(self, sam, uni):
        if sam.shape[-2:] != uni.shape[-2:]:
            uni = F.interpolate(uni, size=sam.shape[-2:], mode='bilinear', align_corners=False)
        a, u = self.sam_projection(sam), self.uni_projection(uni)
        h,w = a.shape[-2:]
        if h % self.window_size or w % self.window_size:
            raise ValueError('Fusion feature dimensions must be divisible by window_size')
        aw,uw = self.windows(a),self.windows(u)
        an,un = self.sam_norm(aw),self.uni_norm(uw)
        da = self.sam_attention(an,un,un,need_weights=False)[0]
        du = self.uni_attention(un,an,an,need_weights=False)[0]
        ap = a + 0.1*self.restore(da,a.shape)
        up = u + 0.1*self.restore(du,u.shape)
        pair = torch.cat((ap,up),dim=1)
        gate = torch.sigmoid(self.channel_gate(F.adaptive_avg_pool2d(pair,1)) + self.spatial_gate(pair))
        fused = gate*ap+(1-gate)*up
        return self.residual_projection(fused), gate, fused
