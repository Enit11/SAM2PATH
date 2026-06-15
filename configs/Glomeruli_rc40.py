from copy import deepcopy

from box import Box

from configs.Glomeruli import config as base_config


config = deepcopy(base_config)
config["opt"]["num_epochs"] = 40

cfg = Box(config)
