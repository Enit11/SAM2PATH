"""Exercise real SamSeg training_step/EMA/save hooks using a small fake encoder.

This is a control-flow test, not a real SAM2/UNI accuracy or memory benchmark.
"""
import sys
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
from pathlib import Path
from copy import deepcopy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from pytorch_lightning import Trainer
from configs.Glomeruli_uni2_bidirectional_dualspace_bsr import cfg as base_cfg
from network.bsr import BoundaryStructureRefinement
from pl_module_sam_seg import SamSeg


class TinyModel(nn.Module):
    geometry_enabled = True

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 256, 1)
        self.refine = BoundaryStructureRefinement(256, 8)
        self.head = nn.Conv2d(256, 4, 1)
        self.distance_head = nn.Conv2d(256, 3, 1)
        self.boundary_head = nn.Conv2d(256, 1, 1)

    def forward(self, images, return_feature=False, return_aux=False):
        f, gate = self.refine(self.features(images))
        logits = self.head(f)
        result = (list(logits), list(logits.mean((-1, -2)).unsqueeze(-1)))
        if return_feature:
            result += (f,)
        if return_aux:
            result += ({'boundary_logits': self.boundary_head(f), 'distance_maps': self.distance_head(f), 'calibration_gate': gate},)
        return result


class TinyData(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        label = torch.zeros(16, 16, dtype=torch.long)
        label[:8, :8] = 1
        label[8:, :8] = 2
        label[:, 8:] = 3
        image = torch.nn.functional.one_hot(label, 4).permute(2, 0, 1).float()[1:]
        return {'labeled': (image, label), 'unlabeled': (image, image)}


def main():
    torch.manual_seed(7)
    torch.set_num_threads(2)
    cfg = deepcopy(base_cfg)
    p = cfg.semi.prototype
    p.warmup_epochs = p.memory.warmup_epochs = cfg.semi.geometry.warmup_epochs = 0
    p.dual_space.memory_match_threshold = 0.
    module = SamSeg(cfg, TinyModel(), MetricCollection({'acc': MulticlassAccuracy(4)}),
                    num_classes=3, ignore_index=None, semi_cfg=cfg.semi,
                    focal_cof=0., ce_cof=1., iou_cof=0.)
    initial = module.model.dual_space_projection.euclidean.weight.detach().clone()
    trainer = Trainer(accelerator='cpu', max_epochs=1, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
    trainer.fit(module, DataLoader(TinyData(), batch_size=1))
    assert module.dual_space.ready[1:].all()
    assert module.dual_space.memory_ready[1:].any()
    assert not module.dual_space.pending
    assert not module.teacher_model.training
    assert not torch.equal(initial, module.model.dual_space_projection.euclidean.weight)
    assert all(p.grad is None for p in module.teacher_model.parameters())
    # SDF sign/channel checks (outside positive; class labels1..3 -> channel0..2).
    label = TinyData()[0]['labeled'][1][None]
    _, sdf, _ = module.geometry_targets(label)
    assert sdf.shape == (1, 3, 16, 16)
    assert sdf[0, 0, 2, 2] < 0 and sdf[0, 0, 12, 12] > 0
    _, absent, _ = module.geometry_targets(torch.zeros_like(label))
    assert (absent == 1).all()
    pred = torch.randn_like(sdf)
    losses = module.supervised_geometry_losses({'boundary_logits': torch.zeros(1, 1, 16, 16), 'distance_maps': pred}, label)
    assert torch.allclose(losses[1], (pred-sdf).abs().mean())
    checkpoint = {'state_dict': module.state_dict(), 'optimizer_states': [1], 'lr_schedulers': [1]}
    module.on_save_checkpoint(checkpoint)
    assert not any(k.startswith('teacher_model.') for k in checkpoint['state_dict'])
    assert 'dual_space.proto_e' in checkpoint['state_dict']
    assert 'optimizer_states' not in checkpoint
    # Reload fixed-shape paired state in a fresh module without real backbones.
    reloaded = SamSeg(cfg, TinyModel(), MetricCollection({'acc': MulticlassAccuracy(4)}),
                    num_classes=3, ignore_index=None, semi_cfg=cfg.semi)
    reloaded.load_state_dict(checkpoint['state_dict'], strict=False)
    assert torch.equal(reloaded.dual_space.proto_e, module.dual_space.proto_e)
    # Verify production warm-start clears paired history, syncs the teacher,
    # and does not restore optimizer state.
    import tempfile
    from main import load_initial_weights
    with tempfile.TemporaryDirectory(prefix='sam2path-bsr-test-') as directory:
        path = Path(directory) / 'tiny.ckpt'
        torch.save(checkpoint, path)
        load_initial_weights(reloaded, str(path))
    assert not reloaded.dual_space.ready.any()
    assert not reloaded.dual_space.memory_ready.any()
    for name, parameter in reloaded.model.named_parameters():
        assert torch.equal(parameter, dict(reloaded.teacher_model.named_parameters())[name])
    from sam2path_inference import GlomeruliInference
    inference = object.__new__(GlomeruliInference)
    inference.cfg = cfg
    raw = label[0].numpy()
    assert (inference.postprocess_prediction(raw) == raw).all()
    print('PASS: real SamSeg 3-batch control flow, optimizer, EMA eval, deferred commits, 3-channel SDF/L1 and checkpoint state')


if __name__ == '__main__':
    main()
