
from simCLRTransforms import SimCLRTrainDataTransform, SimCLREvalDataTransform, SimCLRFinetuneTransform
from utils import  get_mean_std
from torchvision import transforms
import torch
import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder


batch_size = 128

root_dir_train = 'chest_xray/train/'

trainset = ImageFolder(root_dir_train,  transform=SimCLRTrainDataTransform())


trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                         shuffle=True, num_workers=32)

mean, std = get_mean_std(trainloader)

print(mean)
print(std)
