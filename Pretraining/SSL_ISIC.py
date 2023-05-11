from logging import lastResort
from SimCLR import SimCLR
from simCLRTransforms import SimCLRTrainDataTransform, SimCLREvalDataTransform, SimCLRFinetuneTransform
from utils import ISICDataset
from torchvision import transforms
import torch
import os
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl


# Big Batches helps better self supervised learning
batch_size = 160
backbone = "resenet50"
optimizer = "adam"
epochs = 200

mean_ISIC = torch.Tensor([0.6310, 0.5398, 0.5361])
std_ISIC = torch.Tensor([0.2764, 0.2748, 0.2783])
lr_logger = LearningRateMonitor()
callbacks = [lr_logger, RichProgressBar()]

root_dir = 'ISIC_2019_Training_Input/ISIC_2019_Training_Input/'
trainset = ISICDataset(csv_file='ISIC_2019_Training_Input/class_ISIC.csv',
                       root_dir=root_dir, train=True, transforms=SimCLRTrainDataTransform(normalize=transforms.Normalize(mean_ISIC, std_ISIC)))
testset = ISICDataset(csv_file='ISIC_2019_Training_Input/class_ISIC.csv',
                      root_dir=root_dir, train=False, transforms=SimCLREvalDataTransform(normalize=transforms.Normalize(mean_ISIC, std_ISIC)))

trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                         shuffle=True, num_workers=32)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=32)

train_samples = len(trainloader) * batch_size

model = SimCLR(batch_size=batch_size, num_samples=train_samples,
               gpus=3, num_nodes=1, dataset=None, optimizer=optimizer)
trainer = pl.Trainer(callbacks=callbacks, enable_progress_bar=True, accelerator='gpu', devices=3, precision=16, max_epochs=200,
                     fast_dev_run=False, strategy="dp", log_every_n_steps=5, enable_checkpointing=True)
trainer.fit(model, trainloader, testloader)

# takeout the SSL trained ResNet50
pretrained_model = model.encoder

# save checkpoint
checkpoint = {
    "model_state": pretrained_model.state_dict(),
}


torch.save(checkpoint, "SSL_ISIC_state_dict.pth")


torch.save(pretrained_model, "SSL_ISIC.pth")
