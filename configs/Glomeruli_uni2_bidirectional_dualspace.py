"""Bidirectional window interaction, paired prototypes and paired FIFO memory."""
from copy import deepcopy
from box import Box
from configs.Glomeruli_uni2_bidirectional_prototype import config as base_config

config = deepcopy(base_config)
config['name'] = 'glomeruli_uni2_bidirectional_dualspace'
p = config['semi']['prototype']
p.update(enabled=True, geometry='dual', instance_gate_enabled=False,
         region_proto_coef=0.05, intra_region_coef=0.0, sep_coef=0.01,
         sep_margin=0.2, conf_threshold=0.9)
p['dual_space'] = dict(dim=128, euclidean_temperature=0.1,
                       spherical_temperature=0.1, memory_match_threshold=0.6)
p['memory'].update(enabled=True, logits_coef=0.02, feature_coef=0.0,
                   size_per_class=16, use_unlabeled=True)
cfg = Box(config)
