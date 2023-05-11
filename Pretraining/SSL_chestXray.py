from logging import lastResort
from SimCLR import SimCLR
from simCLRTransforms import SimCLRTrainDataTransform, SimCLREvalDataTransform, SimCLRFinetuneTransform
from torchvision import transforms
import torch
import os
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder


batch_size = 128
backbone = "resenet50"
optimizer = "adam"
epochs = 200

mean_chestXray = torch.Tensor([0.5346, 0.5346, 0.5346])
std_chestXray = torch.Tensor([0.2793, 0.2793, 0.2793])
lr_logger = LearningRateMonitor()
callbacks = [lr_logger, RichProgressBar()]


root_dir_train = 'chest_xray/train/'
root_dir_test = 'chest_xray/val/'


trainset = ImageFolder(root_dir_train,  transform=SimCLRTrainDataTransform(
    normalize=transforms.Normalize(mean_chestXray, std_chestXray)))
testset = ImageFolder(root_dir_test, transform=SimCLREvalDataTransform(
    normalize=transforms.Normalize(mean_chestXray, std_chestXray)))

trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                         shuffle=True, num_workers=32)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=32)

train_samples = len(trainloader) * batch_size

model = SimCLR(batch_size=batch_size, num_samples=train_samples,
               gpus=3, num_nodes=1, dataset=None, optimizer=optimizer)
trainer = pl.Trainer(callbacks=callbacks, enable_progress_bar=True, accelerator='gpu', devices=3, precision=16, max_epochs=epochs,
                     fast_dev_run=False, strategy="dp", log_every_n_steps=5, enable_checkpointing=True)
trainer.fit(model, trainloader, testloader)


#take out the resnet50 after SimCLR training.
pretrained_model = model.encoder


# Saving State Dict
checkpoint = {
    "model_state": pretrained_model.state_dict(),
}

torch.save(checkpoint, "SSL_chestXray_state_dict.pth")


# Saving the whiole Model
torch.save(pretrained_model, "SSL_chestXray.pth")
