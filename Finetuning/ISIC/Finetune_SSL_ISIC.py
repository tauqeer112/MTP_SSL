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
import os

batch_size = 128
mean_ISIC = torch.Tensor([0.6310, 0.5398, 0.5361])
std_ISIC = torch.Tensor([0.2764, 0.2748, 0.2783])

pretrained_model = torch.load("pretrained_model_ISIC.pth")


lr_logger = LearningRateMonitor()
checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
callbacks = [lr_logger, RichProgressBar(), checkpoint]

tranform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_ISIC, std=std_ISIC),
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
    ]
)


root_dir = 'ISIC_2019_Training_Input/ISIC_2019_Training_Input/'
trainset = ISICDataset(csv_file='ISIC_2019_Training_Input/class_ISIC.csv',
                       root_dir=root_dir, train=True,
                       transforms=tranform)
testset = ISICDataset(csv_file='ISIC_2019_Training_Input/class_ISIC.csv',
                      root_dir=root_dir, train=False,
                      transforms=tranform)

trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=32)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=32)


model = TransferLearning(8, learning_rate=1e-5)

trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=20, auto_lr_find=False,
                     devices=3, strategy='dp', callbacks=callbacks,
                     enable_progress_bar=True, fast_dev_run=False, max_epochs=100)


trainer.fit(model, trainloader, testloader)

Save_path = os.path.join("SSL_FineTuned_Models","SSL_finetuned_ISIC.pth")
torch.save(model.state_dict(), Save_path)
trainer.test(model, testloader)
