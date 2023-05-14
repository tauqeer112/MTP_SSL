### An Experimental Comparison of SSL, Imagenet Weight, and Random Initialization as Prior Weights for Deep Learning Models
## Authors

- [Tauqeer Akhtar](https://github.com/tauqeer112)
- [Dr. Deepti R. Bathula](https://www.iitrpr.ac.in/deepti-r-bathula)

## Abstract
The use of pre-trained weights from ImageNet
during training on a dissimilar dataset has been shown to result in a better solution convergence than using random weights as a prior. However, medical image datasets have unique characteristics that differ significantly from those in ImageNet, thus making pre-training with self-supervised learning a preferable alternative. By leveraging self-supervised training, the model acquires domain-specific knowledge that can enhance its ability
to perform downstream tasks such as classification segmentation, or detection. Additionally, the model’s performance can be further improved using deep mutual learning or knowledge distillation techniques. We derive the conclusion that Self Supervised Learning prevents the Model 
from overfitting on the training data. It helps training with prior weight initialization, but it still is not outperforming ImageNet in terms of accuracy.

## Datasets and Models

### Datasets
- [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW8)
- [Skin Cancer ISIC](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
- [Covid19](https://www.kaggle.com/datasets/imdevskp/corona-virus-report)
- [Chest X-ray](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Indian Diabetic Retinopathy Image Dataset(IDRiD)](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)

### Models
![Models](https://github.com/tauqeer112/MTP_SSL/blob/main/Images/Screenshot%20from%202023-05-14%2020-43-13.png?raw=true)
- The first model, M1, was directly trained on the dataset without self-supervised learning, and the weights of the ResNet50 architecture were initialized randomly.
- The second model, M2, was trained on the same dataset after self-supervised learning, with the weights randomly initialized before the self-supervised training.
- The third model, M3, was directly trained on the dataset without self-supervised learning, but with pre-trained weights taken from ImageNet.
- The fourth model, M4, was trained on the same dataset after self-supervised learning, with the pre-trained weights from ImageNet assigned before performing the self-supervised learning.

## Results
### Accuracies with different datasets and Models
![Results](https://github.com/tauqeer112/MTP_SSL/blob/main/Images/Results_3rd_se.png?raw=true)

### Accuracy and loss curve while training.

![HAM10000](https://github.com/tauqeer112/MTP_SSL/blob/main/Images/HAM.png?raw=true)
  
![ISIC](https://github.com/tauqeer112/MTP_SSL/blob/main/Images/isic.png?raw=true)

## Conclusions

Our analysis suggests that self-supervised learning is effective in preventing overfitting on the training data and can be helpful for weight initialization. However, we found that self-supervised learning does not necessarily outperform ImageNet in terms of accuracy. Furthermore, we observed that self-supervised learning with ImageNet as the prior performs better than random initialization.

## Reproducing the results

### Requirement

- Pytorch Lightning

### How to run the code:

- There are two folders `Pretraining` and `Finetuning`. The `Pretraining` folder contains SSL code for different datasets which can be run easily. for example `python SSL_covid19.py`
- There are five folders for different datasets for finetuning inside `Finetuning` folder. You can run the python script for random weight initializaion , imageNet as well as SSL after loading the SSL pretrained model.

### How to visualize the results.

- You can use the tensorboard logs stored in lightning logs to visualize the results.