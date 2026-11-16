"""GT distances, holes, CE and feature-calibration gradients; CPU-only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from network.bsr import BoundaryStructureRefinement, boundary_distance_weights, boundary_weighted_ce


def main():
    labels = torch.zeros(1, 16, 16, dtype=torch.long)
    labels[:, 3:13, 3:13] = 3
    labels[:, 6:10, 6:10] = 0  # real GT lumen, never fill
    original = labels.clone()
    weights = boundary_distance_weights(labels)
    assert weights[0, 6, 7] == 5 and weights[0, 0, 0] < 5
    assert torch.equal(original, labels)
    logits = torch.randn(1, 4, 16, 16, requires_grad=True)
    loss = boundary_weighted_ce(logits, labels)
    expected = (torch.nn.functional.cross_entropy(logits, labels, reduction='none') * weights).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(boundary_distance_weights(torch.zeros_like(labels)), torch.ones_like(labels).float())
    ignored = torch.full_like(labels, -100)
    assert boundary_weighted_ce(logits, ignored, ignore_index=-100) == 0
    module = BoundaryStructureRefinement(8, 4)
    x = torch.randn(1, 8, 8, 8, requires_grad=True)
    y, gate = module(x)
    assert torch.equal(x, y)
    y.square().mean().backward()
    assert module.gamma.grad is not None and module.gamma.grad.abs() > 0
    module.gamma.data.fill_(.1)
    module.zero_grad()
    module(x)[0].square().mean().backward()
    assert module.boundary_projection.weight.grad.abs().sum() > 0
    print('PASS: GT holes preserved, exact weighted CE, empty/ignored GT, zero-init identity and calibration gradients')


if __name__ == '__main__':
    main()
