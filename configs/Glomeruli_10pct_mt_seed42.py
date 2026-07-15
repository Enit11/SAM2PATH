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
    "prototype": {
        "enabled": True,
        "warmup_epochs": 5,
        "momentum": 0.99,
        "conf_threshold": 0.85,
        "temperature": 0.1,
        "region_proto_coef": 0.05,
        "intra_region_coef": 0.02,
        "sep_coef": 0.0,
        "min_region_pixels": 6,
        "min_region_purity": 0.7,
        "max_regions_per_image": 16,
        "max_pixels_per_region": 128,
        "use_unlabeled": True,
    },
}

cfg = Box(config)
