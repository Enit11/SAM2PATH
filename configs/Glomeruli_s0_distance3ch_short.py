"""S0: fixed-fusion baseline for the spherical semantic ablation."""

from copy import deepcopy

from box import Box
from configs.Glomeruli_uni2_fusion import config as base_config


config = deepcopy(base_config)
config["name"] = "sam2uni2_s0_distance3ch_short"
config["seed"] = 42
config["check_val_every_n_epoch"] = 1
config["limit_val_batches"] = 1.0
config["opt"]["num_epochs"] = 15
config["model"]["high_res_refinement"]["enabled"] = False
config["semi"]["prototype"]["sep_coef"] = 0.0
config["semi"]["prototype"]["sep_margin"] = 0.2

cfg = Box(config)
