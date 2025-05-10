# Final Project: Garbage Classification with CNN Architectures

## Overview
This repository implements a garbage‐classification pipeline using five different convolutional neural network architectures:

- **VGG**  
- **ResNet**  
- **DenseNet**  
- **MobileNet**  
- **EfficientNet**  

Each model is trained and evaluated on the same garbage dataset so you can compare their performance under identical settings.

---

## Project Structure
```plaintext
final_project/
├── checkpoint/                 # Saved model 
├── dataset/                    # Garbage images, 
├── train/                      # Training scripts for each architecture
│   ├── training_vgg.py
│   ├── training_Resnet.py
│   ├── training_denseNet.py
│   ├── training_MobileNet.py
│   └── training_efficientnet.py
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Download the Dataset

We use the “Garbage Classification” dataset from Kaggle:

🔗 Dataset page: https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data

From the guidance of the page, you can download the data .zip file from the web page.

Then we should unzip the zip file to ``./dataset``. The correct data structure of dataset is:
```
├── dataset/    
│   ├── garbage_classification
```

## Training
After all the data and environment prepared, we can use the script in train to operate the training. We should in the ``./final_project`` file path and then execute following command:

```
$ python ./train/training_denseNet.py
```

## Download Pre‑trained Checkpoints
You can also download the pre-trained checkpoints from the google drive.

For DenseNet, ResNet, EfficientNet and MobileNet: https://drive.google.com/drive/folders/1LVgE_xRWvgKNo6bc-DvNsZ5xTdPLWFoy?usp=drive_link

For VGG checkpoint: https://drive.google.com/drive/u/1/folders/1-INxOohwOFPoxrCYuEjlyfybsPiY8fxo
