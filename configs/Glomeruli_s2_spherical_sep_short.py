"""S2: S0 plus a spherical prototype-separation loss only."""

from copy import deepcopy

from box import Box
from configs.Glomeruli_s0_distance3ch_short import config as base_config


config = deepcopy(base_config)
config["name"] = "sam2uni2_s2_spherical_sep_short"
# Prototype and instance tokens already use L2 normalization. This is the
# only changed loss term relative to S0.
config["semi"]["prototype"]["sep_coef"] = 0.01
config["semi"]["prototype"]["sep_margin"] = 0.2

cfg = Box(config)
