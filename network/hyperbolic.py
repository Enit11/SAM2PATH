"""Float32 Poincare-ball operations; curvature is -c, c > 0."""
import math
import torch


def project(x, c=1.0, eps=1e-5):
    x = x.float()
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return x * ((1.0 - eps) / math.sqrt(c) / norm).clamp(max=1.0)


def exp0(v, c=1.0):
    v = v.float()
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return project(torch.tanh(math.sqrt(c) * norm) * v / (math.sqrt(c) * norm), c)


def log0(x, c=1.0):
    x = project(x, c)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.atanh((math.sqrt(c) * norm).clamp(max=1-1e-5)) * x / (math.sqrt(c) * norm)


def distance2(x, y, c=1.0):
    x, y = project(x, c), project(y, c)
    delta = (x-y).square().sum(dim=-1)
    denom = ((1-c*x.square().sum(dim=-1)) * (1-c*y.square().sum(dim=-1))).clamp_min(1e-8)
    # acosh(1+t)^2 has a finite limit at t=0; this equivalent asinh
    # form avoids infinite derivatives of acosh at coincident points.
    t = (c * delta / denom).clamp_min(1e-12)
    return 4.0 / c * torch.asinh(torch.sqrt(t)).square()
