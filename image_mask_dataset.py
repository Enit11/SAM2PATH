import os
import random

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
try:
    import jpeg4py
except ImportError:
    jpeg4py = None

        
SUB_FOLDER_IMAGE = "img"
SUB_FOLDER_MASK = "mask"


def read_rgb_img(p):
    if p.lower().endswith((".jpg", ".jpeg")) and jpeg4py is not None:
        try:
            return jpeg4py.JPEG(p).decode()
        except:
            return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    else:
        return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def read_mask(p):
    return cv2.imread(p, cv2.IMREAD_UNCHANGED) #.astype(np.int_)


def convert_color_mask(mask, mask_color_map=None, mask_color_mode="rgb"):
    if mask_color_map is None or mask.ndim < 3:
        return mask

    if mask.shape[2] > 3:
        mask = mask[:, :, :3]

    if mask_color_mode.lower() == "rgb":
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    elif mask_color_mode.lower() != "bgr":
        raise ValueError(f"Unsupported mask_color_mode: {mask_color_mode}")

    converted = np.zeros(mask.shape[:2], dtype=np.uint8)
    matched = np.zeros(mask.shape[:2], dtype=bool)

    for color, cls_id in mask_color_map.items():
        if isinstance(color, str):
            color = tuple(int(v) for v in color.split(","))
        color = np.asarray(color, dtype=np.uint8)
        current = np.all(mask == color, axis=-1)
        converted[current] = int(cls_id)
        matched |= current

    if not np.all(matched):
        unknown = np.unique(mask[~matched].reshape(-1, mask.shape[2]), axis=0)
        preview = unknown[:8].tolist()
        raise ValueError(f"Unknown mask colors found: {preview}")

    return converted


def remap_label_mask(mask, label_map=None):
    if label_map is None:
        return mask

    remapped = np.zeros_like(mask)
    for src, dst in label_map.items():
        remapped[mask == int(src)] = int(dst)
    return remapped


class ImageMaskDataset(Dataset):

    def __init__(
            self,
            dataset_root: str,
            dataset_csv_path: str,
            data_type: str,
            val_fold_id: int,
            augmentation=None,
            data_ext: str =".png",
            dataset_mean=(0.485, 0.456, 0.406),
            dataset_std=(0.229, 0.224, 0.225),
            ignored_classes=None, # only supports None, 0 or [0, ...]
            image_folder: str = SUB_FOLDER_IMAGE,
            mask_folder: str = SUB_FOLDER_MASK,
            split_folders: bool = False,
            mask_color_map=None,
            mask_color_mode: str = "rgb",
            label_map=None,
    ):
        super().__init__()

        self.dataset_root = dataset_root
        self.dataset_csv_path = dataset_csv_path
        self.data_ext = data_ext
        self.augmentation = augmentation
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.split_folders = split_folders
        self.mask_color_map = mask_color_map
        self.mask_color_mode = mask_color_mode
        self.label_map = label_map

        self.setup(data_type, val_fold_id)

        self.tensor_transforms = albumentations.Compose([
            albumentations.Normalize(mean=dataset_mean, std=dataset_std),
            ToTensorV2(),

        ])
        self.ignored_classes = ignored_classes

    def __len__(self):
        return len(self.img_list)

    def setup(self, data_type, val_fold_id):
        if data_type not in ['train', 'val', 'test']:
            raise Exception("Not supported dataset type. It should be train, val or test")
        self.data_type = data_type
        self.val_fold_id = val_fold_id
        if data_type == 'test':
            self.val_fold_id = -1

        if val_fold_id >= 0:
            self.img_list = self.read_cv_dataset_csv()
        else:
            if data_type == 'val':
                data_type = 'test'
                self.data_type = data_type
            self.img_list = self.read_dataset_csv()

    def read_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        if self.data_type in ['test']:
            df = df[df['is_test'] > 0]
        else:  # train
            df = df[df['is_test'] == 0]
        return df

    def read_cv_dataset_csv(self):
        df = pd.read_csv(self.dataset_csv_path, header=0)
        if self.data_type in ['val']:
            df = df[df['fold'] == self.val_fold_id]
        elif self.data_type in ['test']:
            df = df[df['fold'] < 0]
        else:
            df = df[df['fold'] > 0]
            df = df[df['fold'] != self.val_fold_id]
        return df

    def process_ignored_classes(self, mask):
        if self.ignored_classes is not None:
            if not isinstance(self.ignored_classes, (list, tuple)):
                self.ignored_classes = [self.ignored_classes]
            for cls in self.ignored_classes:
                if cls != 0:
                    mask[mask == cls] = 0
        return mask

    def __getitem__(self, i):
        row = self.img_list.iloc[i]
        img_id = row['img_id']

        split_root = os.path.join(self.dataset_root, self.data_type) if self.split_folders else self.dataset_root
        image_path = os.path.join(split_root, self.image_folder, img_id + self.data_ext)
        image = read_rgb_img(image_path)
        mask = read_mask(os.path.join(split_root, self.mask_folder, img_id + self.data_ext))
        mask = convert_color_mask(mask, self.mask_color_map, self.mask_color_mode)
        mask = remap_label_mask(mask, self.label_map)

        if self.augmentation is not None:
            ret = self.augmentation(image=image, mask=mask)
            image, mask = ret["image"], ret["mask"]

        mask = self.process_ignored_classes(mask)

        ret = self.tensor_transforms(image=image, mask=mask)
        image, mask = ret["image"], ret["mask"]

        return image, mask.long()

class FtMaskDataset(ImageMaskDataset):
    def __init__(
            self,
            dataset_root: str,
            dataset_csv_path: str,
            data_type: str,
            val_fold_id: int,
            augmentation = None,
            data_ext: str = ".pt", # only changed this
            dataset_mean = (0.485, 0.456, 0.406),
            dataset_std = (0.229, 0.224, 0.225),
            ignored_classes = None, # only supports None, 0 or [0, ...]
            image_folder: str = SUB_FOLDER_IMAGE,
            mask_folder: str = SUB_FOLDER_MASK,
            split_folders: bool = False,
            mask_color_map=None,
            mask_color_mode: str = "rgb",
            label_map=None,
    ):
        super().__init__(
            dataset_root,
            dataset_csv_path,
            data_type,
            val_fold_id,
            augmentation,
            data_ext,
            dataset_mean,
            dataset_std,
            ignored_classes,
            image_folder,
            mask_folder,
            split_folders,
            mask_color_map,
            mask_color_mode,
            label_map,
        )

    def __getitem__(self, i):
        row = self.img_list.iloc[i]
        img_id = row['img_id']

        split_root = os.path.join(self.dataset_root, self.data_type) if self.split_folders else self.dataset_root
        image = torch.load(os.path.join(split_root, self.image_folder, img_id + self.data_ext),
                           map_location='cpu')
        mask = read_mask(os.path.join(split_root, self.mask_folder, img_id + ".png"))
        mask = convert_color_mask(mask, self.mask_color_map, self.mask_color_mode)
        mask = remap_label_mask(mask, self.label_map)

        mask = self.process_ignored_classes(mask)

        mask = torch.from_numpy(mask).long()

        return image, mask


class UnlabeledImageDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            data_ext: str = ".png",
            image_hw=(1024, 1024),
            dataset_mean=(0.485, 0.456, 0.406),
            dataset_std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.image_dir = image_dir
        self.data_ext = data_ext
        self.files = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith(data_ext.lower())
        )
        if not self.files:
            raise ValueError(f"No unlabeled images found in {image_dir}")

        height, width = image_hw
        self.weak_transform = albumentations.Compose([
            albumentations.Resize(height, width),
            albumentations.Normalize(mean=dataset_mean, std=dataset_std),
            ToTensorV2(),
        ])
        self.strong_transform = albumentations.Compose([
            albumentations.Resize(height, width),
            albumentations.ColorJitter(0.2, 0.2, 0.2, 0.08, p=0.8),
            albumentations.GaussNoise(p=0.25),
            albumentations.GaussianBlur(blur_limit=(3, 5), p=0.2),
            albumentations.Normalize(mean=dataset_mean, std=dataset_std),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image = read_rgb_img(os.path.join(self.image_dir, self.files[i]))
        weak = self.weak_transform(image=image)["image"]
        strong = self.strong_transform(image=image)["image"]
        return weak, strong


class SemiSupervisedDataset(Dataset):
    def __init__(self, labeled_dataset: Dataset, unlabeled_dataset: Dataset, unlabeled_ratio: int = 1):
        super().__init__()
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.unlabeled_ratio = max(int(unlabeled_ratio), 1)

    def __len__(self):
        return max(len(self.labeled_dataset), len(self.unlabeled_dataset) * self.unlabeled_ratio)

    def __getitem__(self, i):
        labeled = self.labeled_dataset[i % len(self.labeled_dataset)]
        unlabeled = self.unlabeled_dataset[random.randrange(len(self.unlabeled_dataset))]
        return {
            "labeled": labeled,
            "unlabeled": unlabeled,
        }



class GeneralDataModule(LightningDataModule):
    def __init__(
        self,
        common_cfg_dic,
        dataset_classs,
        cus_transforms,
        batch_size,
        num_workers,
        pin_memory=True,
        persistent_workers=None,
        prefetch_factor=None,
        train_drop_last=False,
        semi_cfg=None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = num_workers > 0 if persistent_workers is None else persistent_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.train_drop_last = train_drop_last
        self.semi_cfg = semi_cfg

        self.dataset_train, self.dataset_val, self.dataset_test = self.initialize_dataset(common_cfg_dic,
                                                                                          dataset_classs,
                                                                                          cus_transforms)
        if self.semi_cfg is not None and self.semi_cfg.get("enabled", False):
            image_dir = self.semi_cfg.get("unlabeled_image_dir", None)
            if image_dir is None:
                image_dir = os.path.join(common_cfg_dic["dataset_root"], "unlabeled", common_cfg_dic["image_folder"])
            if not os.path.isdir(image_dir):
                raise FileNotFoundError(
                    f"Semi-supervised training is enabled, but unlabeled_image_dir does not exist: {image_dir}"
                )
            unlabeled_dataset = UnlabeledImageDataset(
                image_dir=image_dir,
                data_ext=common_cfg_dic["data_ext"],
                image_hw=self.semi_cfg.get("image_hw", (1024, 1024)),
                dataset_mean=common_cfg_dic["dataset_mean"],
                dataset_std=common_cfg_dic["dataset_std"],
            )
            self.dataset_train = SemiSupervisedDataset(
                self.dataset_train,
                unlabeled_dataset,
                unlabeled_ratio=self.semi_cfg.get("unlabeled_ratio", 1),
            )

    def _loader_kwargs(self, shuffle):
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def initialize_dataset(self, common_cfg, DatasetCLS, cus_transforms):
        if cus_transforms is None:
            transforms_train, transforms_eval = None, None
        elif isinstance(cus_transforms, (list, tuple)):
            transforms_train = cus_transforms[0]
            transforms_eval = cus_transforms[1]
        else:
            transforms_train, transforms_eval = cus_transforms, cus_transforms

        dataset_train = DatasetCLS(**common_cfg, data_type="train", augmentation=transforms_train)
        dataset_val = DatasetCLS(**common_cfg, data_type="val", augmentation=transforms_eval)
        dataset_test = DatasetCLS(**common_cfg, data_type="test", augmentation=transforms_eval)
        return dataset_train, dataset_val, dataset_test

    def train_dataloader(self):
        kwargs = self._loader_kwargs(shuffle=True)
        kwargs["drop_last"] = self.train_drop_last
        return DataLoader(self.dataset_train, **kwargs)

    def val_dataloader(self):
        val_kwargs = self._loader_kwargs(shuffle=False)
        val_kwargs["batch_size"] = 1
        return DataLoader(self.dataset_val, **val_kwargs)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, **self._loader_kwargs(shuffle=False))

    def predict_dataloader(self):
        kwargs = self._loader_kwargs(shuffle=False)
        kwargs["batch_size"] = 1
        return DataLoader(self.dataset_test, **kwargs),
