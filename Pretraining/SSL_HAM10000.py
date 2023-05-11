from logging import lastResort
from SimCLR import SimCLR
from simCLRTransforms import SimCLRTrainDataTransform , SimCLREvalDataTransform , SimCLRFinetuneTransform
from utils import HAMDataset
from torchvision import transforms
import torch
import os
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl

batch_size = 160
backbone = "resenet50"
optimizer = "adam" #or "lars"
epochs = 200

mean_HAM = torch.Tensor([0.6650, 0.5399, 0.5497])
std_HAM = torch.Tensor([0.2662, 0.2687, 0.2729])
lr_logger = LearningRateMonitor()
callbacks = [lr_logger,RichProgressBar()]

root_dir = 'ham1000-segmentation-and-classification/images'
trainset = HAMDataset(csv_file='ham1000-segmentation-and-classification/file_label.csv',
                      root_dir = root_dir, train = True , transforms = SimCLRTrainDataTransform(normalize=transforms.Normalize(mean_HAM , std_HAM)))
testset = HAMDataset(csv_file='ham1000-segmentation-and-classification/file_label.csv', 
                     root_dir=root_dir , train=False , transforms= SimCLREvalDataTransform(normalize=transforms.Normalize(mean_HAM , std_HAM)))

trainloader = DataLoader(trainset, batch_size=batch_size,pin_memory=True,
                                          shuffle=True, num_workers=72)
testloader = DataLoader(testset, batch_size=batch_size,
                                          shuffle=False,pin_memory=True, 
                                          num_workers=72)

train_samples =  len(trainloader)* batch_size

model = SimCLR(batch_size=batch_size, num_samples=train_samples,gpus =2, num_nodes=1, dataset =None, optimizer=optimizer)
trainer = pl.Trainer(callbacks=callbacks, enable_progress_bar=True,accelerator='gpu',
                     devices=2,precision=16, max_epochs=200,fast_dev_run=False , 
                     strategy="dp", log_every_n_steps=5 , 
                     enable_checkpointing=True)
trainer.fit(model,trainloader , testloader)

pretrained_model = model.encoder


checkpoint =  {
    "model_state" : pretrained_model.state_dict(),
}

torch.save(checkpoint ,"SSL_HAM10000_state_dict.pth")

torch.save(pretrained_model, "SSL_HAM10000.pth")
