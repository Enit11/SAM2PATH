from argparse import ArgumentParser
import os
import time
import torch
from box import Box
# from lightning.pytorch import seed_everything
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection, JaccardIndex, F1Score
from network.sam_network_2 import PromptSAM, PromptSAMLateFusion, PromptSAMLateFusion_NOKAN
from network.sam_network import PromptSAMLateFusion as PromptSAMLateFusionSAM2
from network.sam_network import PromptSAMLateFusion_NOKAN as PromptSAMLateFusionSAM2NoKAN
# from network.sam_network_backup import PromptSAM, PromptSAMLateFusion
from pl_module_sam_seg import SamSeg
import albumentations
# from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# project_path = "/mnt/zmy/code/sam-path/SAMPath"
# name = "SAM-PATH-Glomeruli"
# tb_writer_summary_path = os.path.join(project_path, "run", name, "Logs")
# current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
# log_dir = os.path.join(tb_writer_summary_path, current_time)
# logger = TensorBoardLogger(log_dir, name='my_model')

# import os
# os.environ["WANDB_MODE"]="offline"


_original_load = torch.load

def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)

torch.load = patched_load
def get_augmentation(cfg):
    # W, H = cfg.dataset.image_hw if cfg.dataset.image_hw is not None else (1024, 1024)
    W, H = cfg.dataset.image_hw
    transform_train_fn = albumentations.Compose([
        albumentations.RandomResizedCrop(size=(H, W), scale=(0.08, 1.0), p=1.0),
        albumentations.HorizontalFlip(p=0.75),
        albumentations.RandomRotate90(),
        albumentations.ColorJitter(0.1, 0.1, 0.1, 0.1),
    ])
    # transform_test_fn = None #albumentations.Compose([])
    transform_test_fn = albumentations.Compose([
        albumentations.Resize(H, W),
    ])
    # transform_train = lambda x: transform_train_fn(image=x[0], mask=x[1])["image"]
    # transform_test = lambda x: transform_test_fn(image=x)["image"]
    # return transform_train, transform_test
    return transform_train_fn, transform_test_fn


def get_metrics(cfg):
    # ``dataset.num_classes`` is the number of foreground classes.  Channel 0
    # is the (supervised) background class.
    num_classes = cfg.dataset.num_classes + 1
    ignore_index = cfg.dataset.ignore_index if "ignore_index" in cfg.dataset else None
    metrics = MetricCollection({
        "IOU_Jaccard_Bal": JaccardIndex(num_classes=num_classes, ignore_index=ignore_index, task='multiclass'),
        "IOU_Jaccard": JaccardIndex(num_classes=num_classes, ignore_index=ignore_index, task='multiclass',
                                        average="micro"),
        "F1": F1Score(num_classes=num_classes, ignore_index=ignore_index, task='multiclass', average="micro"),
        # Pixel-wise Dice is equivalent to F1 for multiclass segmentation.
        "Dice": F1Score(num_classes=num_classes, ignore_index=ignore_index, task='multiclass', average="micro"),
        "Dice_Bal": F1Score(num_classes=num_classes, ignore_index=ignore_index, task='multiclass', average="macro"),
        # Recommended headline score for this experiment: macro Dice over the
        # three foreground classes only.  This avoids background pixels
        # dominating the micro score while still keeping background supervised.
        "Dice_FG_Macro": F1Score(num_classes=num_classes, ignore_index=0, task='multiclass', average="macro"),
        # Vector [background, class_1, class_2, class_3].  SamSeg expands it
        # into scalar TensorBoard tags so each foreground class is inspectable.
        "Dice_Per_Class": F1Score(num_classes=num_classes, task='multiclass', average=None),
    })
    return metrics



def get_model(cfg):
    if cfg.model.extra_encoder is not None:  # cfg.model.extra_encoder  hipt
        print("Using %s as an extra encoder" % cfg.model.extra_encoder)
        # The external visual encoder's projection neck is independent from
        # the SAM decoder/model variant.  Keeping the historical fallback
        # preserves old configs, while new dual-backbone configs can state the
        # required output dimensionality explicitly.
        neck = bool(getattr(cfg.model, "extra_encoder_neck", cfg.model.extra_type == 'plus'))
        if cfg.model.extra_encoder == 'hipt':
            from network.get_network import get_hipt
            extra_encoder = get_hipt(cfg.model.extra_checkpoint, neck=neck)
        elif cfg.model.extra_encoder == 'uni_v1': # sim added
            from network.get_network import get_uni
            extra_encoder = get_uni(
                cfg.model.extra_checkpoint,
                neck=neck,
                trainable_neck=bool(getattr(cfg.model, "extra_encoder_trainable_neck", False)),
            )
            print("\n========== UNI DEBUG ==========")
            print("extra_type:", cfg.model.extra_type)
            print("neck:", neck)
            print("extra_encoder:", type(extra_encoder))
            print("input_dim:", getattr(extra_encoder, "input_dim", None))
            print("output_dim:", getattr(extra_encoder, "output_dim", None))
            print("model:", type(extra_encoder.model))
            print("================================\n")
        elif cfg.model.extra_encoder == 'uni2_h':
            from network.get_network import get_uni2_h
            extra_encoder = get_uni2_h(
                cfg.model.extra_checkpoint,
                neck=neck,
                trainable_neck=bool(getattr(cfg.model, "extra_encoder_trainable_neck", False)),
            )
        else:
            raise NotImplementedError
    else:
        extra_encoder = None
    if cfg.model.extra_type in ['plus']:
        MODEL = PromptSAM
    elif cfg.model.extra_type in ['fusion']:
        MODEL = PromptSAMLateFusion
    elif cfg.model.extra_type in ['sam2_fusion']:
        MODEL = PromptSAMLateFusionSAM2
    elif cfg.model.extra_type in ['sam2_fusion_nokan']:
        MODEL = PromptSAMLateFusionSAM2NoKAN
    else:
        raise NotImplementedError

    model_kwargs = {
        "model_type": cfg.model.type,
        "checkpoint": cfg.model.checkpoint,
        "prompt_dim": cfg.model.prompt_dim,
        "num_classes": cfg.dataset.num_classes,
        "extra_encoder": extra_encoder,
        "freeze_image_encoder": cfg.model.freeze.image_encoder,
        "freeze_prompt_encoder": cfg.model.freeze.prompt_encoder,
        "freeze_mask_decoder": cfg.model.freeze.mask_decoder,
        "mask_HW": cfg.dataset.image_hw,
        "feature_input": cfg.dataset.feature_input,
        "prompt_decoder": cfg.model.prompt_decoder,
        "dense_prompt_decoder": cfg.model.dense_prompt_decoder,
        "no_sam": cfg.model.no_sam if "no_sam" in cfg.model else None,
        "extra_encoder_dim": getattr(extra_encoder, "output_dim", None),
    }
    if MODEL is PromptSAMLateFusionSAM2NoKAN:
        model_kwargs["geometry_cfg"] = cfg.model.geometry if "geometry" in cfg.model else None
        model_kwargs["high_res_refinement_cfg"] = (
            cfg.model.high_res_refinement if "high_res_refinement" in cfg.model else None
        )
        model_kwargs["spherical_fusion_cfg"] = (
            cfg.model.spherical_fusion if "spherical_fusion" in cfg.model else None
        )
    model = MODEL(
        **model_kwargs,
    )
    return model

def get_data_module(cfg):
    from image_mask_dataset import GeneralDataModule, ImageMaskDataset, FtMaskDataset
    augs = get_augmentation(cfg)
    common_cfg_dic = {
        "dataset_root": cfg.dataset.dataset_root,
        "dataset_csv_path": cfg.dataset.dataset_csv_path,
        "val_fold_id": cfg.dataset.val_fold_id,
        "data_ext": ".jpg" if "data_ext" not in cfg.dataset else cfg.dataset.data_ext,
        "dataset_mean": cfg.dataset.dataset_mean,
        "dataset_std": cfg.dataset.dataset_std,
        "ignored_classes": cfg.dataset.ignored_classes,  # only supports None, 0 or [0, ...]
        "image_folder": cfg.dataset.image_folder if "image_folder" in cfg.dataset else "img",
        "mask_folder": cfg.dataset.mask_folder if "mask_folder" in cfg.dataset else "mask",
        "split_folders": cfg.dataset.split_folders if "split_folders" in cfg.dataset else False,
        "mask_color_map": cfg.dataset.mask_color_map if "mask_color_map" in cfg.dataset else None,
        "mask_suffix": cfg.dataset.mask_suffix if "mask_suffix" in cfg.dataset else "",
        "binary_foreground": bool(cfg.dataset.binary_foreground) if "binary_foreground" in cfg.dataset else False,
        "mask_color_mode": cfg.dataset.mask_color_mode if "mask_color_mode" in cfg.dataset else "rgb",
        "label_map": cfg.dataset.label_map if "label_map" in cfg.dataset else None,
        "train_label_column": cfg.dataset.train_label_column if "train_label_column" in cfg.dataset else None,
        "train_labeled_value": cfg.dataset.train_labeled_value if "train_labeled_value" in cfg.dataset else 1,
    }
    if cfg.dataset.feature_input is True:
        dataset_cls = FtMaskDataset
    else:
        dataset_cls = ImageMaskDataset

    data_module = GeneralDataModule(common_cfg_dic, dataset_cls, cus_transforms=augs,
                                    batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                                    pin_memory=cfg.pin_memory if "pin_memory" in cfg else True,
                                    persistent_workers=cfg.persistent_workers if "persistent_workers" in cfg else None,
                                    prefetch_factor=cfg.prefetch_factor if "prefetch_factor" in cfg else None,
                                    train_drop_last=cfg.train_drop_last if "train_drop_last" in cfg else False,
                                    semi_cfg=cfg.semi if "semi" in cfg else None)
    return data_module



def get_pl_module(cfg, model, metrics):
    pl_module = SamSeg(
        cfg = cfg,
        sam_model = model,
        metrics = metrics,
        num_classes = cfg.dataset.num_classes,
        focal_cof = cfg.loss.focal_cof,
        dice_cof = cfg.loss.dice_cof,
        ce_cof=cfg.loss.ce_cof,
        iou_cof = cfg.loss.iou_cof,
        rc_cof=cfg.loss.rc_cof if "rc_cof" in cfg.loss else 0.0,
        rc_sigma=cfg.loss.rc_sigma if "rc_sigma" in cfg.loss else 0.5,
        class_weights=cfg.loss.class_weights if "class_weights" in cfg.loss else None,
        focal_gamma=cfg.loss.focal_gamma if "focal_gamma" in cfg.loss else 2.0,
        lr = cfg.opt.learning_rate,
        weight_decay = cfg.opt.weight_decay,
        lr_steps =  cfg.opt.steps,
        warmup_steps=cfg.opt.warmup_steps,
        ignored_index=cfg.dataset.ignored_classes_metric,
        # Configurations without this key use class 0 as a supervised tissue class.
        # Legacy foreground-only configurations keep their explicit ignore_index=0.
        ignore_index=cfg.dataset.ignore_index if "ignore_index" in cfg.dataset else None,
        semi_cfg=cfg.semi if "semi" in cfg else None,
    )
    return pl_module

def load_initial_weights(pl_module, init_ckpt_path):
    if init_ckpt_path is None:
        return
    checkpoint = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    # Warm-start explicitly discards all data-dependent historical banks.
    # It must not silently retain paired state while replacing the EMA teacher.
    for key in list(state_dict):
        if key.startswith("dual_space."):
            state_dict.pop(key)
    if getattr(pl_module, "dual_space", None) is not None:
        pl_module.dual_space.reset()
    checkpoint_has_geometry = any(
        key.startswith("model.boundary_head.") or key.startswith("model.distance_head.")
        for key in state_dict
    )
    model_geometry_enabled = getattr(pl_module.model, "geometry_enabled", None)
    if model_geometry_enabled is not None and checkpoint_has_geometry != model_geometry_enabled:
        raise ValueError(
            "Checkpoint and config build different model architectures: "
            f"checkpoint_has_geometry={checkpoint_has_geometry}, "
            f"config_geometry_enabled={model_geometry_enabled}. "
            "Use the configuration that created this checkpoint."
        )
    # Instance memory is run-specific pseudo-instance state; rebuild it from the new teacher.
    if getattr(pl_module, "proto_geometry", "cosine") == "poincare":
        # Cosine prototypes have unit norm and cannot be interpreted as
        # Poincare coordinates. Rebuild from the loaded EMA teacher.
        state_dict.pop("foundation_prototypes", None)
        state_dict.pop("foundation_proto_ready", None)
    for key in ("instance_memory", "instance_memory_ready", "instance_memory_ptr"):
        state_dict.pop(key, None)
    if "foundation_prototypes" in state_dict and hasattr(pl_module, "foundation_prototypes"):
        loaded_proto = state_dict["foundation_prototypes"]
        if loaded_proto.ndim == 2 and loaded_proto.shape[0] == pl_module.num_classes + 1:
            pl_module.foundation_prototypes = loaded_proto.new_zeros(loaded_proto.shape)
    if "foundation_proto_ready" in state_dict and hasattr(pl_module, "foundation_proto_ready"):
        loaded_ready = state_dict["foundation_proto_ready"]
        if loaded_ready.ndim == 1 and loaded_ready.shape[0] == pl_module.num_classes + 1:
            pl_module.foundation_proto_ready = loaded_ready.new_zeros(loaded_ready.shape)
    target_state = pl_module.state_dict()
    incompatible = {
        key: (tuple(value.shape), tuple(target_state[key].shape))
        for key, value in state_dict.items()
        if key in target_state and value.shape != target_state[key].shape
    }
    if incompatible:
        print("[INFO] Skipping incompatible checkpoint tensors:")
        for key, (source_shape, target_shape) in incompatible.items():
            print(f"  {key}: checkpoint {source_shape} -> model {target_shape}")
            state_dict.pop(key)
    missing, unexpected = pl_module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing checkpoint keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {unexpected}")
    # Lightweight checkpoints intentionally omit the EMA teacher.  After the
    # student weights have been loaded, initialize the teacher from the same
    # student state rather than leaving the deepcopy created in SamSeg.__init__
    # at its pre-checkpoint values.  This is a warm start, not an exact resume.
    if getattr(pl_module, "teacher_model", None) is not None:
        teacher_missing, teacher_unexpected = pl_module.teacher_model.load_state_dict(
            pl_module.model.state_dict(), strict=False
        )
        if teacher_missing:
            print(f"[WARN] Teacher sync missing keys: {teacher_missing}")
        if teacher_unexpected:
            print(f"[WARN] Teacher sync unexpected keys: {teacher_unexpected}")
        print("[INFO] EMA teacher initialized from loaded student weights (warm start).")


def main(cfg, ckpt_path=None, init_ckpt_path=None):
    if ckpt_path is not None or init_ckpt_path is not None:
        # PyTorch 2.6+ loads Lightning checkpoints with weights_only=True by default.
        # Our checkpoints include Box config objects, so we allowlist Box for resume.
        torch.serialization.add_safe_globals([Box])

    if "seed" in cfg:
        seed_everything(int(cfg.seed), workers=True)
    torch.backends.cudnn.benchmark = bool(cfg.benchmark) if "benchmark" in cfg else True
    
    # sim added: logger module
    project_path = cfg.out_dir
    name = cfg.name
    tb_writer_summary_path = os.path.join(project_path, "run", name, "Logs")
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_dir = os.path.join(tb_writer_summary_path, current_time)
    logger = TensorBoardLogger(log_dir, name="sam2path")

    
    data_module = get_data_module(cfg)

    sam_model = get_model(cfg)

    metrics = get_metrics(cfg=cfg)

    pl_module = get_pl_module(cfg, model=sam_model, metrics=metrics)  # 包装给 pl

    load_initial_weights(pl_module, init_ckpt_path)

    # logger = WandbLogger(project=cfg.project, name=cfg.name, save_dir=cfg.out_dir, log_model=True)
    

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # The Lightning default checkpoint contains student + EMA teacher + AdamW
    # moments.  For this 1.5B-parameter dual-network setup that can exhaust
    # host RAM during serialization.  SamSeg.on_save_checkpoint strips the
    # redundant teacher and optimizer states; keep a single latest lightweight
    # weights checkpoint per run.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        save_weights_only=True,
        filename="epoch={epoch:03d}-step={step}",
    )

    accumulate_grad_batches = cfg.accumulate_grad_batches if "accumulate_grad_batches" in cfg else 1

    trainer = Trainer(default_root_dir=cfg.out_dir, logger=logger,
                      devices=cfg.devices,
                      max_epochs=cfg.opt.num_epochs,
                      accelerator="gpu", #strategy="auto",
                      #strategy='ddp_find_unused_parameters_true',
                      log_every_n_steps=cfg.log_every_n_steps if "log_every_n_steps" in cfg else 20,
                      num_sanity_val_steps=0,
                      precision=cfg.opt.precision,
                      callbacks=[lr_monitor, checkpoint_callback],
                      accumulate_grad_batches=accumulate_grad_batches,
                      check_val_every_n_epoch=cfg.check_val_every_n_epoch if "check_val_every_n_epoch" in cfg else 1,
                      limit_val_batches=cfg.limit_val_batches if "limit_val_batches" in cfg else 0.5,
                      fast_dev_run=False)

    trainer.fit(pl_module, data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    
# python main.py --config configs.Glomeruli --devices 0 --project sampath --name glomeruli_run0
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(False)
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument('--devices', type=lambda s: [int(item) for item in s.split(',')], default=[1])
    parser.add_argument('--project', type=str, default="sim-sampath-crag")
    parser.add_argument('--name', type=str, default="crag_run4_kan_sam")
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--init_ckpt_path', type=str, default=None)
    # parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    module = __import__(args.config, globals(), locals(), ['cfg'])
    cfg = module.cfg

    cfg["project"] = args.project
    cfg["devices"] = args.devices
    cfg["name"] = args.name
    # cfg["seed"] = args.seed

    # seed_everything(cfg["seed"])
    print(cfg)
    main(cfg, ckpt_path=args.ckpt_path, init_ckpt_path=args.init_ckpt_path)
    # print(cfg)
