from box import Box


config = {
    "batch_size": 3,
    "num_workers": 6,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "train_drop_last": True,
    "benchmark": True,
    "out_dir": "outputs/glomeruli",
    "check_val_every_n_epoch": 5,
    "log_every_n_steps": 20,
    "opt": {
        "num_epochs": 80,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "precision": "bf16-mixed",
        "steps": [5900, 6900],
        "warmup_steps": 200,
    },
    "model": {
        "type": "vit_b",
        "checkpoint": "",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": False,
            "mask_decoder": False,
        },
        "prompt_dim": 256,
        "prompt_decoder": False,
        "dense_prompt_decoder": False,
        "extra_encoder": None,
        "extra_type": "sam2_fusion_nokan",
        "extra_checkpoint": None,
    },
    "loss": {
        "focal_cof": 0.125,
        "dice_cof": 0.875,
        "ce_cof": 0.25,
        "iou_cof": 0.0,
        "rc_cof": 0.08,
        "rc_sigma": 0.5,
    },
    "dataset": {
        "dataset_root": "dataset",
        "dataset_csv_path": "dataset_cfg/glomeruli_cv.csv",
        "data_ext": ".png",
        "val_fold_id": -1,
        "num_classes": 4,
        "ignored_classes": 0,
        "ignored_classes_metric": None,
        "ignore_index": 0,
        "image_hw": (1024, 1024),
        "feature_input": False,
        "image_folder": "image",
        "mask_folder": "mask",
        "split_folders": True,
        "mask_color_mode": "bgr",
        "mask_color_map": {
            (0, 0, 0): 0,        # background
            (46, 254, 100): 1,   # glomerulus
            (0, 0, 255): 2,      # tubule, red in the source mask
            (255, 0, 0): 3,      # vessel, blue in the source mask
            (255, 0, 170): 0,    # unlabeled/other
            (88, 250, 244): 0,   # unlabeled/other
        },
        "dataset_mean": (0.485, 0.456, 0.406),
        "dataset_std": (0.229, 0.224, 0.225),
    },
}

cfg = Box(config)
