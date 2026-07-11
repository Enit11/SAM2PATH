from copy import deepcopy
from box import Box
from configs.Glomeruli import config as base_config

config = deepcopy(base_config)
config["dataset"]["dataset_csv_path"] = "dataset_cfg/glomeruli_10pct_seed42.csv"
config["opt"]["num_epochs"] = 100
config["batch_size"] = 2
config["num_workers"] = 2
config["persistent_workers"] = False
config["prefetch_factor"] = 2
config["semi"] = {
    "enabled": True,
    "unlabeled_source_csv_path": "dataset_cfg/glomeruli_cv.csv",
    "exclude_labeled_from_unlabeled": True,
    "unlabeled_ratio": 1,
    "ema_decay": 0.999,
    "consistency_coef": 1.0,
    "warmup_epochs": 10,
    "conf_threshold": 0.8,
    "loss_type": "mse",
    "image_hw": (1024, 1024),
}

cfg = Box(config)
