import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from pytorch_lightning import LightningModule


def get_mean_std(loader):
    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        print("current batch = ", num_batches)
        data = data[0]
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squares_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squares_sum / num_batches - mean**2)**0.5

    return mean, std


class HAMDataset(Dataset):
    def __init__(self, csv_file, root_dir, train=True, transforms=None):
        self.df = pd.read_csv(csv_file, index_col=0)
        if(train == True):
            self.df = self.df.iloc[:7000, :]
        else:
            self.df = self.df.iloc[7000:, :]
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name_image = self.df.iloc[idx, 0]
        image_filepath = os.path.join(self.root_dir, name_image)
        image = Image.open(image_filepath)
        if self.transforms:
            image = self.transforms(image)

        label = self.df.iloc[idx, -1]

        return image, label


class ISICDataset(Dataset):
    def __init__(self, csv_file, root_dir, train=True, transforms=None):
        self.df = pd.read_csv(csv_file, index_col=0)
        if(train == True):
            train_sp = int(0.8 * len(self.df))
            self.df = self.df.iloc[:train_sp, :]
        else:
            train_sp = int(0.8 * len(self.df))
            self.df = self.df.iloc[train_sp:, :]
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name_image = self.df.iloc[idx, 0]
        image_filepath = os.path.join(self.root_dir, name_image)
        image = Image.open(image_filepath)
        if self.transforms:
            image = self.transforms(image)

        label = self.df.iloc[idx, -1]

        return image, label


class RetinoPathyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.df = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name_image = self.df.iloc[idx, 0]
        image_filepath = os.path.join(self.root_dir, name_image)
        image = Image.open(image_filepath+'.jpg')
        if self.transforms:
            image = self.transforms(image)

        label = self.df.iloc[idx, -1]

        return image, label


class TransferLearning(LightningModule):
    def __init__(self, numofclass, learning_rate):
        super().__init__()
        self.numofclass = numofclass
        # init a pretrained resnet
        backbone = pretrained_model
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        # self.criterion = criterion
        self.learning_rate = learning_rate

        # use the pretrained model to classify cifar-10 (10 image classes)
        self.classifier = nn.Sequential(nn.Linear(num_filters, 512, bias=True),
                                        nn.Linear(512, 256, bias=True),
                                        nn.Linear(256, numofclass, bias=True),
                                        )

    def forward(self, x):
        self.feature_extractor.eval()

        if(self.current_epoch > 10):
            representations = self.feature_extractor(x).flatten(1)
        else:
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)

        x = self.classifier(representations)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):

        # Output from Dataloader
        imgs, labels = batch

        # Prediction
        preds = self.forward(imgs)
        # Calc Loss
        loss = F.cross_entropy(preds, labels)

        # Calc accuracy
        _, preds = torch.max(preds, 1)
        accuracy = torch.sum(preds == labels).float() / preds.size(0)

        logs = {'train_loss': loss, 'train_accuracy': accuracy}
        self.log_dict(logs)

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)

    # Validation Loop

    def validation_step(self, batch, batch_idx):
        '''
        OPTIONAL
        SAME AS "trainning_step"
        '''
        # Output from Dataloader
        imgs, labels = batch

        # Prediction
        preds = self.forward(imgs)
        # Calc Loss
        loss = F.cross_entropy(preds, labels)

        # Calc accuracy
        _, preds = torch.max(preds, 1)
        accuracy = torch.sum(preds == labels).float() / preds.size(0)

        logs = {'val_loss': loss, 'val_accuracy': accuracy}
        self.log_dict(logs)

        return {'val_loss': loss, 'val_accuracy': accuracy, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        '''
        OPTIONAL
        SAME AS "trainning_step"
        '''
        # Output from Dataloader
        imgs, labels = batch

        # Prediction
        preds = self.forward(imgs)
        # Calc Loss
        loss = F.cross_entropy(preds, labels)

        # Calc accuracy
        _, preds = torch.max(preds, 1)
        accuracy = torch.sum(preds == labels).float() / preds.size(0)

        logs = {'test_loss': loss, 'test_accuracy': accuracy}
        self.log_dict(logs)

        # return logs

        return {'test_loss': loss, 'test_accuracy': accuracy, 'log': logs, 'progress_bar': logs}

    # Aggegate Validation Result
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        logs = {'avg_val_loss': avg_loss, 'avg_val_accuracy': avg_accuracy}
        torch.cuda.empty_cache()

        return {'avg_val_loss': avg_loss, 'log': logs}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['test_accuracy']
                                   for x in outputs]).mean()
        logs = {'avg_test_loss': avg_loss, 'avg_test_accuracy': avg_accuracy}
        torch.cuda.empty_cache()

        return {'avg_test_loss': avg_loss, "avg_test_accuracy": avg_accuracy, 'log': logs}
