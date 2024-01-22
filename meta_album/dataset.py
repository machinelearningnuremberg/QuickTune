#Taken from https://github.com/ihsaan-ullah/meta-album/blob/master/Code/DataLoader/data_loader_standard.py

import os
import glob
import json
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Tuple, List, Any
DATA_FRAME = pd.core.frame.DataFrame
AVAILABLE_MTLBM_DATASETS = {"set0": ["BCT", "BRD", "CRS", "FLW", "MD_MIX", "PLK", "PLT_VIL", "RESISC", "SPT", "TEX"],
                            "set1": ["ACT_40", "APL", "DOG", "INS_2", "MD_5_BIS","MED_LF", "PLT_NET", "PNU", "RSICB", "TEX_DTD"],
                            "set2": ["ACT_410", "AWA", "BTS", "FNG", "INS", "MD_6", "PLT_DOC", "PRT", "RSD", "TEX_ALOT"]}
AVAILABLE_SETS = ["set0", "set1", "set2"]
DEFAULT_FOLDER = "datasets/meta-album"

class StandardDataset(Dataset):
    """
    PyTorch dataset with the field "targets"
    """

    def __init__(self, 
                 dataset: str, 
                 transform: Any = None, 
                 image_mode: str = "RGB",
                 val_size: int = 0.2,
                 test_size: int = 0.2,
                 split: str = "train",
                 seed: int = 93) -> None:
        super().__init__()

        self.dataset_directory = dataset
        self.image_mode = image_mode
        self.transform = transform
        self.test_size = test_size
        self.val_size = val_size
        assert os.path.exists(
            self.dataset_directory), f"Dataset path {self.dataset_directory} not found."

        self.items = self.construct_items()
        self.add_field_targets()

        random_gen = np.random.default_rng(seed)
        self.nb_classes = int(max(self.targets)) + 1
        self.idxs = np.arange(len(self.items))
        random_gen.shuffle(self.idxs)
        split_val_idx = int(len(self.items)*self.test_size)
        split_test_idx = int(len(self.items)*(self.test_size+self.val_size))
        if split == "train":
            idxs = self.idxs[split_test_idx:]
        elif split == "val":
            idxs = self.idxs[:split_val_idx]
        elif split == "test":
            idxs = self.idxs[split_val_idx:split_test_idx]
        else:
            raise ValueError("Not implemented split.")

        self.items = [item for i, item in enumerate(self.items) if i in idxs.tolist()]

    def __len__(self) -> int:
        return len(self.items)

    def get_image_size(self):
        img_path, _, label = self.items[0]
        img = Image.open(img_path).convert(self.image_mode)
        return img.size

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        img_path, _, label = self.items[idx]
        img = Image.open(img_path).convert(self.image_mode)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def construct_items(self) -> List[list]:
        # Get raw labels
        self.read_info_json()
        df = self.read_labels_csv()

        search_pattern = os.path.join(self.dataset_directory, "images", "*.*")
        items = list()

        # Get list of lists, each list contains the image path and the label
        for path in sorted(glob.glob(search_pattern)):
            base_path = os.path.basename(path)
            file_ext = os.path.splitext(base_path)[-1].lower()
            if file_ext in [".png", ".jpg", ".jpeg"]:
                info = [path, df.loc[base_path, self.category_column_name]]
                items.append(info)

        # Map each raw label to a label (int starting from 0)
        self.raw_label2label = dict()
        for item in items:
            if item[1] not in self.raw_label2label:
                self.raw_label2label[item[1]] = len(self.raw_label2label)
            item.append(self.raw_label2label[item[1]])
        return items

    def read_info_json(self) -> None:
        info_json_path = os.path.join(self.dataset_directory, "info.json")
        with open(info_json_path, "r") as f:
            info_json = json.load(f)
        # "FILE_NAME"
        self.image_column_name = info_json["image_column_name"]
        # "CATEGORY"
        self.category_column_name = info_json["category_column_name"]

    def read_labels_csv(self) -> DATA_FRAME:
        csv_path = os.path.join(self.dataset_directory, "labels.csv")
        df = pd.read_csv(csv_path, sep=",", encoding="utf-8")
        df = df.loc[:, [self.image_column_name, self.category_column_name]]
        df.set_index(self.image_column_name, inplace=True)
        return df

    def add_field_targets(self) -> None:
        """
        The targets field is available in nearly all torchvision datasets. It 
        must be a list containing the label for each data point (usually the y 
        value).
        
        https://avalanche.continualai.org/how-tos/avalanchedataset/creating-avalanchedatasets
        """
        self.targets = [item[2] for item in self.items]
        self.targets = torch.tensor(self.targets, dtype=torch.int64)

    def get_raw_label2label_dict(self) -> dict:
        return self.raw_label2label