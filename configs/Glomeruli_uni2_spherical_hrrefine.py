"""Formal SAM2PATH training: spherical SAM2/UNI fusion + high-resolution refinement.

This configuration is intentionally separate from the baseline and all short
screening configurations.  It can be initialized from an existing geometry
checkpoint with --init_ckpt_path; new spherical/refinement layers are safely
zero-initialized residual paths.
"""
from copy import deepcopy

from box import Box

from configs.Glomeruli_uni2_fusion import config as base_config


config = deepcopy(base_config)
config["name"] = "glomeruli_uni2_spherical_hrrefine"
config["seed"] = 42
config["check_val_every_n_epoch"] = 1
config["limit_val_batches"] = 1.0
config["opt"]["num_epochs"] = 100
# Windows/8-GB GPU safe loader settings.  The previous run reached the end of
# epoch 0 and failed while serializing the giant EMA checkpoint; single-process
# loading also removes worker/pinned-batch memory pressure while diagnosing it.
config["num_workers"] = 0
config["pin_memory"] = False
config["persistent_workers"] = False
config["prefetch_factor"] = None
config["model"]["extra_encoder_neck"] = True
config["model"]["extra_encoder_trainable_neck"] = True

# Keep the validated SAM2 + UNI2 concat neck, then learn a zero-initialized
# residual correction from a unit-sphere, spatially gated fusion stream.
config["model"]["spherical_fusion"] = {
    "enabled": True,
    "projection_channels": 256,
    "residual_scale": 0.10,
    "eps": 1e-6,
}

# Use the frozen SAM2 backbone FPN at its highest available resolution to
# correct the four decoder logits.  Its residual prediction also begins at 0.
config["model"]["high_res_refinement"] = {
    "enabled": True,
    "hidden_channels": 64,
    "residual_scale": 0.10,
}

# Labelled-only supervision for background gaps between two distinct connected
# components of the same semantic class.  It is intentionally not generated
# from teacher pseudo-labels: the diagnosed failure is a same-class merge.
# The direct main-logit term is the primary corrective signal; the auxiliary
# separator head provides a boundary-specific diagnostic representation.
geometry = config["semi"]["geometry"]
geometry.update({
    "separator_enabled": True,
    "separator_radius": 6,
    "separator_min_instance_pixels": 6,
    "separator_head_coef": 0.05,
    "separator_logit_coef": 0.10,
    # The existing implementation compares labelled student geometry with
    # unlabelled teacher geometry.  They are different images, so do not
    # optimise that invalid pairwise term.
    "unsup_coef": 0.0,
    "feature_calibration_enabled": True,
    "feature_calibration_scale": 0.10,
    "boundary_consistency_coef": 0.05,
})
# ``model.geometry`` constructs auxiliary heads, while ``semi.geometry``
# supplies their losses; keep the construction flag explicit in both places.
config["model"]["geometry"]["separator_enabled"] = True

# Teacher-derived class prototypes and reliable instance memory live on the
# unit hypersphere (F.normalize in the Lightning module).  GT always updates
# them; unlabeled updates require confidence >= 0.90.
prototype = config["semi"]["prototype"]
prototype.update({
    "enabled": True,
    "warmup_epochs": 5,
    "momentum": 0.99,
    "conf_threshold": 0.90,
    "temperature": 0.10,
    "region_proto_coef": 0.05,
    "intra_region_coef": 0.02,
    "sep_coef": 0.01,
    "sep_margin": 0.20,
    "min_region_pixels": 6,
    "min_region_purity": 0.70,
    "max_regions_per_image": 16,
    "max_pixels_per_region": 128,
    "use_unlabeled": True,
    "unlabeled_weight": 0.50,
    "instance_gate_enabled": True,
    "instance_gate_coef": 0.05,
})
prototype["memory"].update({
    "enabled": True,
    "warmup_epochs": 10,
    "size_per_class": 16,
    "momentum": 0.90,
    "conf_threshold": 0.90,
    "min_region_pixels": 32,
    "min_region_purity": 0.90,
    "temperature": 0.10,
    "consistency_coef": 0.05,
    "unlabeled_weight": 0.50,
    "use_unlabeled": True,
})

cfg = Box(config)
