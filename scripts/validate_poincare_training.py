"""One real-data GPU optimisation step with every auxiliary branch enabled."""
import os
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pytorch_lightning import Trainer, Callback, seed_everything
from configs.Glomeruli_uni2_hyperbolic_prototype import cfg
from main import get_data_module, get_model, get_metrics, get_pl_module, load_initial_weights


class CheckGradients(Callback):
    def on_after_backward(self, trainer, module):
        gradients = [(n, p.grad) for n, p in module.named_parameters() if p.grad is not None]
        bad = [n for n, g in gradients if not torch.isfinite(g).all()]
        if bad:
            raise RuntimeError(f"Nonfinite gradients: {bad}")
        if not gradients:
            raise RuntimeError("No gradients produced")
        if not module.foundation_proto_ready[1:].any():
            raise RuntimeError("No prototypes updated")
        radius = module.foundation_prototypes.norm(dim=-1).max().item()
        if radius >= 1 / module.proto_curvature ** 0.5:
            raise RuntimeError("Prototype outside Poincare ball")
        print(f"[PASS] {len(gradients)} finite gradients; max prototype radius={radius:.6f}", flush=True)


if __name__ == "__main__":
    seed_everything(42, workers=True)
    cfg.batch_size = 1
    cfg.num_workers = 0
    cfg.semi.prototype.warmup_epochs = 0
    cfg.semi.prototype.memory.warmup_epochs = 0
    cfg.semi.geometry.warmup_epochs = 0
    data = get_data_module(cfg)
    module = get_pl_module(cfg, get_model(cfg), get_metrics(cfg))
    load_initial_weights(module, str(Path(__file__).resolve().parents[1] / "outputs/glomeruli/run/glomeruli_uni2_spherical_hrrefine_warmstart/Logs/20261013-150230/sam2path/version_0/checkpoints/last.ckpt"))
    trainer = Trainer(accelerator="gpu", devices=[0], precision=cfg.opt.precision,
                      max_steps=1, limit_train_batches=1, limit_val_batches=0,
                      num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
                      enable_model_summary=False, callbacks=[CheckGradients()])
    trainer.fit(module, datamodule=data)
    print("[PASS] Real-data Poincare training step completed", flush=True)
