"""GT-only boundary weighting and jointly trained residual feature calibration."""
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class BoundaryStructureRefinement(nn.Module):
    def __init__(self, channels=256, boundary_channels=64, gamma_init=0.):
        super().__init__()
        self.boundary_projection = nn.Conv2d(channels, boundary_channels, 3, padding=1)
        self.gate = nn.Conv2d(boundary_channels, 1, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, structure, boundary=None):
        # The predicted gate is a learned modulation, NOT a GT boundary mask.
        attention = torch.sigmoid(self.gate(F.gelu(self.boundary_projection(structure))))
        return structure + self.gamma * attention * structure, attention


@torch.no_grad()
def boundary_distance_weights(labels, strength=4., sigma=3., ignore_index=None):
    if sigma <= 0 or strength < 0:
        raise ValueError('BSR sigma must be positive and strength nonnegative')
    arrays = labels.detach().cpu().numpy()
    results = []
    for label in arrays:
        valid = np.ones_like(label, dtype=bool) if ignore_index is None else label != ignore_index
        edge = np.zeros_like(label, dtype=bool)
        diff = (label[1:] != label[:-1]) & valid[1:] & valid[:-1]
        edge[1:] |= diff
        edge[:-1] |= diff
        diff = (label[:, 1:] != label[:, :-1]) & valid[:, 1:] & valid[:, :-1]
        edge[:, 1:] |= diff
        edge[:, :-1] |= diff
        weights = np.ones_like(label, dtype=np.float32)
        if edge.any():
            distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            weights += strength * np.exp(-distance / sigma)
        weights[~valid] = 0.
        results.append(weights)
    return torch.as_tensor(np.stack(results), device=labels.device)


def boundary_weighted_ce(logits, labels, strength=4., sigma=3., ignore_index=None):
    valid = torch.ones_like(labels, dtype=torch.bool) if ignore_index is None else labels != ignore_index
    if not valid.any():
        return logits.sum() * 0
    weights = boundary_distance_weights(labels, strength, sigma, ignore_index)
    loss = F.cross_entropy(logits.float(), labels, reduction='none', ignore_index=-100 if ignore_index is None else ignore_index)
    # Divide by valid pixel count, NOT weight sum: matches requested formula.
    return (weights * loss)[valid].mean()
