from copy import deepcopy

from box import Box

from configs.Glomeruli import config as base_config


config = deepcopy(base_config)
config["name"] = "glomeruli_uni2_fusion"
config["opt"]["num_epochs"] = 100
config["model"]["type"] = "vit_b"
config["model"]["checkpoint"] = r"C:\Users\ASUS\SAM2PATH\checkpoints\sam_vit_b_01ec64.pth"
config["model"]["freeze"]["image_encoder"] = True
config["model"]["freeze"]["prompt_encoder"] = False
config["model"]["freeze"]["mask_decoder"] = False
config["model"]["prompt_decoder"] = False
config["model"]["dense_prompt_decoder"] = False
config["model"]["extra_encoder"] = "uni2_h"
config["model"]["extra_type"] = "sam2_fusion_nokan"
config["model"]["extra_checkpoint"] = r"C:\Users\ASUS\SAM2PATH\uni2_h"
config["model"]["no_sam"] = False
config["model"]["geometry"] = {
    "enabled": True,
    "multi_scale_adapter": True,
    "head_channels": 64,
    "residual_scale": 0.1,
}
config["model"]["high_res_refinement"] = {
    # Enable only in the dedicated high-resolution-refinement ablation.
    # The zero-initialized residual preserves the original decoder output at
    # initialization, while learning from the frozen high-resolution SAM2 FPN.
    "enabled": False,
    "hidden_channels": 64,
    "residual_scale": 0.1,
}

config["semi"]["geometry"] = {
    "enabled": True,
    "warmup_epochs": 10,
    "boundary_coef": 0.20,
    "distance_coef": 0.10,
    "hard_ce_coef": 0.10,
    "unsup_coef": 0.05,
    "broadcast_coef": 0.02,
    "boundary_width": 3,
    "sdf_truncation": 32.0,
    "entropy_threshold": 0.45,
    "margin_threshold": 2.0,
    "boundary_threshold": 0.35,
    "teacher_conf_threshold": 0.80,
    "vessel_boundary_weight": 2.5,
}
config["semi"]["enabled"] = True
config["semi"]["unlabeled_source_csv_path"] = "dataset_cfg/glomeruli_split_10pct_seed42.csv"
config["semi"]["exclude_labeled_from_unlabeled"] = True
config["semi"]["labeled_filter_column"] = "labeled"
config["semi"]["labeled_filter_value"] = 1
config["semi"]["prototype"]["memory"]["enabled"] = True

cfg = Box(config)
