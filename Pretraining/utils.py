import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


def get_mean_std(loader):
    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        print("current batch = ", num_batches)
        data = data[0]
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squares_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squares_sum / num_batches - mean**2)**0.5

    return mean, std


class HAMDataset(Dataset):
    def __init__(self, csv_file, root_dir, train=True, transforms=None):
        self.df = pd.read_csv(csv_file, index_col=0)
        if(train == True):
            self.df = self.df.iloc[:7000, :]
        else:
            self.df = self.df.iloc[7000:, :]
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name_image = self.df.iloc[idx, 0]
        image_filepath = os.path.join(self.root_dir, name_image)
        image = Image.open(image_filepath)
        if self.transforms:
            image = self.transforms(image)

        label = self.df.iloc[idx, -1]

        return image, label


class ISICDataset(Dataset):
    def __init__(self, csv_file, root_dir, train=True, transforms=None):
        self.df = pd.read_csv(csv_file, index_col=0)
        if(train == True):
            train_sp = int(0.8 * len(self.df))
            self.df = self.df.iloc[:train_sp, :]
        else:
            train_sp = int(0.8 * len(self.df))
            self.df = self.df.iloc[train_sp:, :]
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name_image = self.df.iloc[idx, 0]
        image_filepath = os.path.join(self.root_dir, name_image)
        image = Image.open(image_filepath)
        if self.transforms:
            image = self.transforms(image)

        label = self.df.iloc[idx, -1]

        return image, label


class RetinoPathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.df = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name_image = self.df.iloc[idx, 0]
        image_filepath = os.path.join(self.root_dir, name_image)
        image = Image.open(image_filepath+'.jpg')
        if self.transforms:
            image = self.transforms(image)

        label = self.df.iloc[idx, -1]

        return image, label
