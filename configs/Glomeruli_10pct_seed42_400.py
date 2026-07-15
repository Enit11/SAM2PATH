from copy import deepcopy
from box import Box
from configs.Glomeruli import config as base_config

config = deepcopy(base_config)
config["dataset"]["dataset_csv_path"] = "dataset_cfg/glomeruli_10pct_seed42.csv"
config["opt"]["num_epochs"] = 400

cfg = Box(config)
