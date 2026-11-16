"""Full dual-space training with three-component GT-supervised BSR."""
from copy import deepcopy
from box import Box
from configs.Glomeruli_uni2_bidirectional_dualspace import config as base_config

config = deepcopy(base_config)
config['name'] = 'glomeruli_uni2_bidirectional_dualspace_bsr'
config['inference'] = dict(preserve_raw_semantics=True)
config['model']['geometry'].update(bsr_enabled=True, bsr_gamma_init=0.0,
                                  feature_calibration_enabled=True)
g = config['semi']['geometry']
g.update(distance_coef=0.10, hard_ce_coef=0.0, unsup_coef=0.0,
         sdf_truncation=32.0, vessel_boundary_weight=1.0)
g['bsr'] = dict(enabled=True, weighted_ce_coef=0.1, weight_strength=4.0, sigma_pixels=3.0)
p = config['semi']['prototype']
p['min_valid_fraction'] = 0.7
p['dual_space']['unlabeled_update_weight'] = 0.5
p['memory'].update(min_region_pixels=32, min_valid_fraction=0.9, conf_threshold=0.95)
# BSR does not define a morphological postprocess. Raw predictions retain
# background lumina/holes according to their original semantic GT.
cfg = Box(config)
