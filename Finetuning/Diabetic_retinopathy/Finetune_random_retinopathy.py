import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from utils import RetinoPathyDataset, TransferLearning
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
import os


batch_size = 16
mean_retinopathy = torch.Tensor([0.4746, 0.3126, 0.2146])
std_retinopathy = torch.Tensor([0.2818, 0.2516, 0.2050])

pretrained_model = resnet50(pretrained=False)

lr_logger = LearningRateMonitor()
checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
callbacks = [lr_logger, RichProgressBar(), checkpoint]

tranform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_retinopathy, std=std_retinopathy),
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
    ]

)


root_dir_train = 'DiseaseGrading/Original_Images/Training_Set/'
root_dir_test = 'DiseaseGrading/Original_Images/Testing_Set/'
trainset = RetinoPathyDataset(csv_file='DiseaseGrading/final_train.csv',
                              root_dir=root_dir_train,
                              transforms=tranform)
testset = RetinoPathyDataset(csv_file='DiseaseGrading/final_test.csv',
                             root_dir=root_dir_test,
                             transforms=tranform)

trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                         shuffle=True, num_workers=32)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, pin_memory=True, num_workers=32)

model = TransferLearning(5, learning_rate=1e-4)

trainer = pl.Trainer(accelerator="gpu", log_every_n_steps=20, auto_lr_find=False,
                     devices=3, strategy='dp', callbacks=callbacks,
                     enable_progress_bar=True, fast_dev_run=False, max_epochs=100)

trainer.fit(model, trainloader, testloader)


Save_path = os.path.join("SSL_FineTuned_Models","Finetuned_random_retinopathy.pth")

torch.save(model.state_dict(), Save_path)
trainer.test(model, testloader)
