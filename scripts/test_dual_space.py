"""CPU numerical/gradient smoke test; no foundation checkpoints required."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from network.dual_space_prototype import DualSpaceProjection, PairedPrototypeMemory


def main():
    torch.manual_seed(42)
    projection = DualSpaceProjection(4, 8)
    teacher = copy.deepcopy(projection).requires_grad_(False)
    bank = PairedPrototypeMemory(3, 8, 2, {'memory_match_threshold': 0.0})
    owner = SimpleNamespace(model=SimpleNamespace(dual_space_projection=projection),
        teacher_model=SimpleNamespace(dual_space_projection=teacher),
        proto_conf_threshold=.9, proto_momentum=.9, proto_sep_margin=.2)
    owner.proto_cfg = {'min_valid_fraction': .7}
    owner.memory_cfg = {'min_region_pixels': 2, 'min_valid_fraction': .9, 'conf_threshold': .95}
    owner.proto_min_region_pixels = 2
    owner.proto_max_regions_per_image = 16
    owner.resize_labels_for_feature = lambda x, hw: x
    owner.resize_conf_for_feature = lambda x, hw: x
    owner.iter_regions = lambda x, v: [(int(((x == c) & v).sum()), c, ((x == c) & v).numpy())
        for c in (1, 2, 3) if ((x == c) & v).any()]
    x = torch.randn(1, 4, 4, 4, requires_grad=True)
    labels = torch.ones(1, 4, 4, dtype=torch.long)
    labels[:, 2:] = 2
    bank.update(owner, x.detach(), labels)
    assert bank.ready.tolist() == [False, True, True, False]
    bank.update(owner, x.detach(), labels, memory=True)
    losses = bank.losses(owner, x, labels)
    memory = bank.losses(owner, x, labels, memory=True)
    total = losses[0] + .01*losses[2] + memory[0]
    total.backward()
    for name, parameter in projection.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None for p in teacher.parameters())
    before = bank.proto_e.clone()
    bank.update(owner, x.detach(), labels, valid_map=torch.zeros_like(labels, dtype=torch.bool))
    assert torch.equal(before, bank.proto_e)
    assert not bank.memory_ready[0].any()
    assert bank.memory_ready.any()
    assert torch.allclose(bank.proto_s[bank.ready].norm(dim=-1), torch.ones(2), atol=1e-5)
    # Duplicating one class's identical memory entries must not increase its mass.
    e, s = bank.proto_e[1], bank.proto_s[1]
    be, bs = bank.proto_e[1:3], bank.proto_s[1:3]
    original = bank.scores(e, s, be, bs, projection, torch.tensor([1, 2])).exp()
    duplicate = bank.scores(e, s, be[[0, 0, 1]], bs[[0, 0, 1]], projection, torch.tensor([1, 1, 2])).exp()
    assert torch.allclose(original[0], duplicate[:2].sum(), atol=1e-6)
    # Pending writes leave pre-batch state unchanged; commit only after loss.
    bank.defer_writes = True
    before = bank.proto_e.clone()
    bank.update(owner, x.detach() + 1, labels)
    bank.update(owner, x.detach(), labels, memory=True)
    assert torch.equal(before, bank.proto_e)
    bank.flush(owner)
    assert not bank.pending and not bank.defer_writes
    assert not torch.equal(before, bank.proto_e)
    # The memory-specific threshold is stricter than prototype confidence.
    conf = torch.full_like(labels, .92, dtype=torch.float)
    assert len(list(bank.regions(owner, x, labels, projection, conf))) == 2
    assert len(list(bank.regions(owner, x, labels, projection, conf, memory=True))) == 0
    bank.reset()
    assert not bank.ready.any() and not bank.memory_ready.any()
    print('PASS: paired banks, missing/empty classes, finite projection/gate/input gradients, detached teacher')


if __name__ == '__main__':
    main()
