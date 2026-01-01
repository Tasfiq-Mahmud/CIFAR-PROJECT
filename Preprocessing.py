#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.models import efficientnet_b0
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim import Adam
from sklearn.metrics import confusion_matrix
import sys

torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu" # detect the GPU if any, if not use CPU, change cuda to mps if you have a mac
print("Device available: ", device)


# In[2]:


BASE_DIR = Path.cwd()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
sys.path


# In[3]:


from Training import ModelTrainer, show_missclassification, give_predictions


# In[4]:


path = BASE_DIR/'Dataset/cifar-10/'

data_df = pd.read_csv(path/'trainLabels.csv',delimiter= ',')
data_df


# In[5]:


if 'id' in data_df.columns:
    data_df['path']=str(path)+'/train/'+data_df['id'].astype(str)+'.png'
    data_df.drop(columns='id',inplace=True)
data_df['path'][0]


# In[6]:


sample_df = data_df.sample(frac=1,random_state=29).reset_index(drop=True) # shuffle the dataframe
sample_df


# In[7]:


data_df.label.value_counts()


# In[8]:


train_df = sample_df.sample(frac = 0.9).reset_index(drop=True)
val_df = sample_df.drop(train_df.index).reset_index(drop=True)
val_df.label.value_counts()


# In[9]:


mean=[0.4911, 0.4821, 0.4465]
std=[0.2470, 0.2434, 0.2616]

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]
)

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=32,padding=4),
        # transforms.RandomGrayscale(p=0.1),
        # transforms.RandomRotation(degrees=15),
        # transforms.RandomApply(
        #     [transforms.ColorJitter(0.1, 0.1, 0.1, 0.02)],
        #     p=0.5
        # ),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]
)

class_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(size=32,padding=4),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]
)


# In[10]:


le = LabelEncoder()
le.fit(sample_df['label'])


# In[11]:
aug_classes = le.transform(['dog','cat','deer','horse'])

class CifarDataset(Dataset):
    def __init__(self, dataframe, transform = None, class_transform = None):
        super().__init__()
        self.img_path = list(dataframe['path'])
        if type(dataframe['label'][0])==np.int64:
            self.labels = dataframe['label']
        else:
            self.labels = torch.tensor(le.transform(dataframe['label']))
        self.transform = transform
        self.class_transform = class_transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        path = self.img_path[index]
        img = Image.open(path).convert('RGB')
        if self.class_transform and self.labels[index] in aug_classes:
            img = self.class_transform(img)
        elif self.transform:
            img = self.transform(img)
        return img, self.labels[index]



# In[12]:


train_dataset = CifarDataset(train_df,train_transform,class_transform)
val_dataset = CifarDataset(val_df, transform)


# In[13]:


BATCH_SIZE = 128
EPOCHS = 70
LR = 1e-3
N_Models = 1


# In[14]:


train_data = DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle = True)
val_data = DataLoader(val_dataset,batch_size=BATCH_SIZE, shuffle=True)

file_name = 'EfficientNetModelTweakedwithaug'