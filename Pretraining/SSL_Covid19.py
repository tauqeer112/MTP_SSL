from logging import lastResort
from SimCLR import SimCLR
from simCLRTransforms import SimCLRTrainDataTransform , SimCLREvalDataTransform , SimCLRFinetuneTransform
from utils import HAMDataset , get_mean_std,ISICDataset, RetinoPathyDataset
from torchvision import transforms
import torch
import os
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder


batch_size = 32
backbone = "resenet50"
optimizer = "adam"
epochs = 500

mean_covid19 = torch.Tensor([0.4957, 0.4957, 0.4957])
std_covid19 = torch.Tensor([0.3009, 0.3009, 0.3009])
lr_logger = LearningRateMonitor()
callbacks = [lr_logger,RichProgressBar()]



root_dir_train = 'Covid19-dataset/train/'
root_dir_test = 'Covid19-dataset/test/'


trainset = ImageFolder(root_dir_train,  transform = SimCLRTrainDataTransform(normalize=transforms.Normalize(mean_covid19 , std_covid19)))
testset = ImageFolder(root_dir_test , transform= SimCLREvalDataTransform(normalize=transforms.Normalize(mean_covid19 , std_covid19)))

trainloader = DataLoader(trainset, batch_size=batch_size,pin_memory=True,
                                          shuffle=True, num_workers=16)
testloader = DataLoader(testset, batch_size=batch_size,
                                          shuffle=False,pin_memory=True, num_workers=16)

train_samples =  len(trainloader)* batch_size

model = SimCLR(batch_size=batch_size, num_samples=train_samples,gpus =3, num_nodes=1, dataset =None, optimizer=optimizer)
trainer = pl.Trainer(callbacks=callbacks, enable_progress_bar=True,accelerator='gpu',
                     devices=3,precision=16, max_epochs=epochs,fast_dev_run=False , strategy="dp", log_every_n_steps=5 , enable_checkpointing=True)
trainer.fit(model,trainloader , testloader)

pretrained_model = model.encoder

checkpoint =  {
    "model_state" : pretrained_model.state_dict(),
}

torch.save(checkpoint ,"checkpoint_covid19.pth")

torch.save(pretrained_model, "SSL_covid19.pth")
