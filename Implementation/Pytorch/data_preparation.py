#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import requests
import io
import numpy as np  
from datetime import date, timedelta
import re
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
#from github import Github
#import github
import torch
import torch.nn as nn
# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader
# Import nn.functional
import torch.nn.functional as F
import torch.optim as optim
from typing import Union, Tuple
import os
import sys
import time
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
import math
import random
import imageio
#from sklearn.metrics import mean_absolute_percentage_error
matplotlib.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, response_variable_index=0, number_feature = 6):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back),:number_feature]
    dataX.append(a)
    dataY.append(dataset[i + look_back, response_variable_index])
  return np.array(dataX), np.array(dataY)


# In[4]:


def data_preparation(df, scaling_range=(0,1),time_step=5,number_feature=6, response_variable_index=3,data_split_ratio=0.8,Suffle=True):
    df = df.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=scaling_range)
    dataset = scaler.fit_transform(df.copy())
    X, Y = create_dataset(dataset, time_step,response_variable_index=response_variable_index, number_feature=number_feature)
    # split into train and test sets
    train_size = int(len(dataset) * data_split_ratio)
    test_size = len(dataset) - train_size
    trainX, testX = X[0:train_size,:], X[train_size:len(dataset),:]
    trainY, testY = Y[0:train_size], Y[train_size:len(dataset)]
    print(trainX.shape)
    # reshape input to be [samples, time steps, features]
    if not multi_feature:
      trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
      testX = np.reshape(testX, (testX.shape[0], testX.shape[1],1))
    #print(trainX.shape)
    X_train=trainX
    X_test=testX
    y_train=trainY.reshape(-1,1)

    print(X_train.shape, y_train.shape)
    # summarize the data
    inputs = torch.from_numpy(X_train)
    targets = torch.from_numpy(y_train)
    # Define dataset
    train_ds = TensorDataset(inputs, targets)

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=Suffle)

    y_test=testY.reshape(-1,1)

    inputs = torch.from_numpy(X_test)
    targets = torch.from_numpy(y_test)
    # Define dataset
    #test_ds = TensorDataset(inputs, targets)
    test_ds=(inputs, targets)
    #test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_ds,scaler


# In[ ]:




