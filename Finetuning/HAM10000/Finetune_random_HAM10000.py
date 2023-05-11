import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from utils import HAMDataset, TransferLearning
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
import os

batch_size = 64
mean_HAM = torch.Tensor([0.6650, 0.5399, 0.5497])
std_HAM = torch.Tensor([0.2662, 0.2687, 0.2729])

pretrained_model = resnet50(pretrained=False)


mean_HAM = torch.Tensor([0.6650, 0.5399, 0.5497])
std_HAM = torch.Tensor([0.2662, 0.2687, 0.2729])
lr_logger = LearningRateMonitor()
checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
callbacks = [lr_logger, RichProgressBar(), checkpoint]

tranform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_HAM, std=std_HAM),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.2),
    ]
)

root_dir = 'ham1000-segmentation-and-classification/images'
trainset = HAMDataset(csv_file='ham1000-segmentation-and-classification/file_label.csv',
                      root_dir=root_dir, train=True, transforms=tranform)
testset = HAMDataset(csv_file='ham1000-segmentation-and-classification/file_label.csv',
                     root_dir=root_dir, train=False, transforms=tranform)

trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                         shuffle=True, num_workers=64)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=64)


model = TransferLearning(7, learning_rate=0.00001)

trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=20, auto_lr_find=False,
                     devices=3, strategy='dp', callbacks=callbacks,
                     enable_progress_bar=True, fast_dev_run=False, max_epochs=50)

trainer.fit(model, trainloader, testloader)

Save_path = os.path.join("SSL_FineTuned_Models","Finetuned_random_HAM10000.pth")

torch.save(model.state_dict(), Save_path)
trainer.test(model, testloader)
