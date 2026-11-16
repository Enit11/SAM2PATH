"""Complete bidirectional fusion + cosine region prototype experiment."""
from copy import deepcopy
from box import Box
from configs.Glomeruli_uni2_spherical_hrrefine import config as base_config

config = deepcopy(base_config)
config['name'] = 'glomeruli_uni2_bidirectional_prototype'
config['model']['spherical_fusion'] = {
    'enabled': True, 'mode': 'bidirectional', 'window_size': 8,
    'heads': 4, 'residual_scale': 0.1,
}
config['semi']['prototype'].update({'enabled': True, 'geometry': 'cosine', 'instance_gate_enabled': False})
config['model']['geometry']['feature_calibration_enabled'] = True
config['model']['geometry']['feature_calibration_scale'] = 0.1
config['semi']['geometry']['unsup_coef'] = 0.0
cfg = Box(config)
