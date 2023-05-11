import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from utils import ISICDataset, TransferLearning
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
import os

batch_size = 128
mean_chestXray = torch.Tensor([0.5346, 0.5346, 0.5346])
std_chestXray = torch.Tensor([0.2793, 0.2793, 0.2793])

pretrained_model = torch.load("SSL_chestXray.pth")

lr_logger = LearningRateMonitor()
checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
callbacks = [lr_logger, RichProgressBar(), checkpoint]

tranform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_chestXray, std=std_chestXray),
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
    ]
)


root_dir_train = 'chest_xray/train/'
root_dir_test = 'chest_xray/val/'

trainset = ImageFolder(root_dir_train,  transform=tranform)
testset = ImageFolder(root_dir_test, transform=tranform)

trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                         shuffle=True, num_workers=16)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=16)


model = TransferLearning(2, learning_rate=1e-5)

trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=20, auto_lr_find=False,
                     devices=3, strategy='dp', callbacks=callbacks,
                     enable_progress_bar=True, fast_dev_run=False, max_epochs=100)

trainer.fit(model, trainloader, testloader)
Save_path = os.path.join("SSL_FineTuned_Models","SSL_finetuned_chestXray.pth")

torch.save(model.state_dict(), Save_path)
trainer.test(model, testloader)
