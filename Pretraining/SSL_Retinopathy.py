from logging import lastResort
from SimCLR import SimCLR
from simCLRTransforms import SimCLRTrainDataTransform, SimCLREvalDataTransform, SimCLRFinetuneTransform
from utils import RetinoPathyDataset
from torchvision import transforms
import torch
import os
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl

batch_size = 32
backbone = "resenet50"
optimizer = "adam"
epochs = 200

mean_retinopathy = torch.Tensor([0.4746, 0.3126, 0.2146])
std_retinopathy = torch.Tensor([0.2818, 0.2516, 0.2050])
lr_logger = LearningRateMonitor()
callbacks = [lr_logger, RichProgressBar()]


root_dir_train = 'DiseaseGrading/Original_Images/Training_Set/'
root_dir_test = 'DiseaseGrading/Original_Images/Testing_Set/'
trainset = RetinoPathyDataset(csv_file='DiseaseGrading/final_train.csv',
                              root_dir=root_dir_train,
                              transforms=SimCLRTrainDataTransform(normalize=transforms.Normalize(mean_retinopathy, std_retinopathy)))
testset = RetinoPathyDataset(csv_file='DiseaseGrading/final_test.csv',
                             root_dir=root_dir_test,
                             transforms=SimCLREvalDataTransform(normalize=transforms.Normalize(mean_retinopathy, std_retinopathy)))

trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                         shuffle=True, num_workers=32)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, pin_memory=True,
                        num_workers=32)

train_samples = len(trainloader) * batch_size

model = SimCLR(batch_size=batch_size, num_samples=train_samples, gpus=3,
               num_nodes=1, dataset=None, optimizer=optimizer)
trainer = pl.Trainer(callbacks=callbacks, enable_progress_bar=True,
                     accelerator='gpu', devices=3, precision=16, max_epochs=epochs,
                     fast_dev_run=False, strategy="dp", log_every_n_steps=5,
                     enable_checkpointing=True)
trainer.fit(model, trainloader, testloader)

# Taking out the resnet50
pretrained_model = model.encoder

# Saving the model state dict
checkpoint = {
    "model_state": pretrained_model.state_dict(),
}
torch.save(checkpoint, "SSL_Retinopathy_state_dict.pth")


# saving the whole model
torch.save(pretrained_model, "SSL_Retinopathy.pth")
