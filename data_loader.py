import os
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn

class TrainDataset(Dataset):

    def __init__(self, data: List[str], classname, target_size, train_transform = lambda x: x, cutpaste = False):
        """
        Loads images from data

        @param data:
            paths to images
        @param: target_size: tuple (int, int), default: (128, 128)
            the desired output size
        """
        super(TrainDataset, self).__init__()
        self.target_size = (target_size[0], target_size[1])
        self.data = data
        self.classname = classname
        self.train_val_split = 0
        self.split = 0
        self.seed = 0
        self.augment = 0
        self.train_transform = train_transform
        self.cutpaste = cutpaste
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.data[idx]).convert('RGB')
        # Pad to square
        img = transforms.Pad(((img.height - img.width) // 2, 0), fill=0)(img)
        # Resize
        img = img.resize(self.target_size, Image.BICUBIC)        
        # CutPaste Transform
        if self.cutpaste == True:
            _, img, gt = self.train_transform(img)
        # Convert to tensor
        else:
            img = (transforms.ToTensor()(img))
            # img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            return img
        # print(img.size)

        # img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img, gt


class TrainDataModule(pl.LightningDataModule):
    def __init__(self, split_dir: str, target_size, batch_size: int = 32, train_transform = lambda x: x, cutpaste = False):
        """
        Data module for training
        @param split_dir: str
            path to directory containing the split files
        @param: target_size: tuple (int, int), default: (128, 128)
            the desired output size
        @param: batch_size: int, default: 32
            batch size
        """
        super(TrainDataModule, self).__init__()
        self.train_transform = train_transform
        self.target_size = (target_size[0], target_size[1])
        print(self.target_size)
        self.input_size = (3, target_size[0], target_size[1])
        self.batch_size = batch_size
        self.name = "training mri images"
        self.cutpaste = cutpaste
        
        train_csv_ixi = os.path.join(split_dir, 'ixi_normal_train.csv')
        train_csv_fastMRI = os.path.join(split_dir, 'normal_train.csv')
        val_csv = os.path.join(split_dir, 'normal_val.csv')

        # Load csv files
        train_files_ixi = pd.read_csv(train_csv_ixi)['filename'].tolist()
        train_files_fastMRI = pd.read_csv(train_csv_fastMRI)['filename'].tolist()
        val_files = pd.read_csv(val_csv)['filename'].tolist()

        # Combine files
        self.train_data = train_files_ixi + train_files_fastMRI
        self.val_data = val_files

        # Logging
        print(f"Using {len(train_files_ixi)} IXI images "
              f"and {len(train_files_fastMRI)} fastMRI images for training. "
              f"Using {len(val_files)} images for validation.")

    def train_dataloader(self):
        return DataLoader(TrainDataset(self.train_data,classname= "",target_size= self.target_size, train_transform=self.train_transform, cutpaste= self.cutpaste),
                          batch_size=self.batch_size,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(TrainDataset(self.val_data,classname="", target_size=self.target_size, train_transform=self.train_transform, cutpaste = self.cutpaste),
                          batch_size=self.batch_size,
                          shuffle=False)


class TestDataset(Dataset):

    def __init__(self, img_csv: str, pos_mask_csv: str, neg_mask_csv: str, target_size):
        """
        Loads anomalous images, their positive masks and negative masks from data_dir

        @param img_csv: str
            path to csv file containing filenames to the images
        @param img_csv: str
            path to csv file containing filenames to the positive masks
        @param img_csv: str
            path to csv file containing filenames to the negative masks
        @param: target_size: tuple (int, int), default: (128, 128)
            the desired output size
        """
        super(TestDataset, self).__init__()
        self.target_size = (target_size[0], target_size[1])
        self.img_paths = pd.read_csv(img_csv)['filename'].tolist()
        self.pos_mask_paths = pd.read_csv(pos_mask_csv)['filename'].tolist()
        self.neg_mask_paths = pd.read_csv(neg_mask_csv)['filename'].tolist()

        assert len(self.img_paths) == len(self.pos_mask_paths) == len(self.neg_mask_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = img.resize(self.target_size, Image.BICUBIC)
        img = transforms.ToTensor()(img)
        # img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # Load positive mask
        pos_mask = Image.open(self.pos_mask_paths[idx])
        pos_mask = pos_mask.resize(self.target_size, Image.NEAREST)
        pos_mask = transforms.ToTensor()(pos_mask)

        # Load negative mask
        neg_mask = Image.open(self.neg_mask_paths[idx])
        neg_mask = neg_mask.resize(self.target_size, Image.NEAREST)
        neg_mask = transforms.ToTensor()(neg_mask)

        return img, pos_mask, neg_mask


def get_test_dataloader(split_dir: str, pathology: str, target_size: Tuple[int, int], batch_size: int):
    """
    Loads test data from split_dir

    @param split_dir: str
        path to directory containing the split files
    @param pathology: str
        pathology to load
    @param batch_size: int
        batch size
    """
    img_csv = os.path.join(split_dir, f'{pathology}.csv')
    pos_mask_csv = os.path.join(split_dir, f'{pathology}_ann.csv')
    neg_mask_csv = os.path.join(split_dir, f'{pathology}_neg.csv')

    return DataLoader(TestDataset(img_csv, pos_mask_csv, neg_mask_csv, target_size),
                      batch_size=batch_size,
                      shuffle=False,
                      drop_last=False)


def get_all_test_dataloaders(split_dir: str, target_size: Tuple[int, int], batch_size: int):
    """
    Loads all test data from split_dir

    @param split_dir: str
        path to directory containing the split files
    @param batch_size: int
        batch size
    """
    pathologies = [
        'absent_septum',
        'artefacts',
        'craniatomy',
        'dural',
        'ea_mass',
        'edema',
        'encephalomalacia',
        'enlarged_ventricles',
        'intraventricular',
        'lesions',
        'mass',
        'posttreatment',
        'resection',
        'sinus',
        'wml',
        'other'
    ]
    return {pathology: get_test_dataloader(split_dir, pathology, target_size, batch_size)
            for pathology in pathologies}