"""Poincare prototype ablation; encoder fusion and memory retain baseline geometry."""
from copy import deepcopy
from box import Box
from configs.Glomeruli_uni2_spherical_hrrefine import config as base_config

config = deepcopy(base_config)
config["name"] = "glomeruli_uni2_hyperbolic_prototype"
config["semi"]["prototype"].update({
    "geometry": "poincare",
    "curvature": 1.0,
    "tangent_scale": 0.25,
    "hyperbolic_margin": 1.0,
})
cfg = Box(config)
