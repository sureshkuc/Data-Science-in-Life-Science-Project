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


# In[4]:


def fit(
    model: nn.Module, 
    optimizer: optim.Optimizer, criterion: nn,
    data: Union[DataLoader, Tuple[DataLoader]], 
    max_epochs: int, 
    cuda=True):
  use_test = False
  if isinstance(data, DataLoader):
    train_loader = data
  elif isinstance(data, tuple):
    if len(data) == 2:
      train_loader, test_loader = data
      if not isinstance(train_loader, DataLoader):
        raise TypeError(f'Expected 1st entry of type DataLoader, but got {type(train_loader)}!')
      #if not isinstance(test_loader, DataLoader):
       # raise TypeError(f'Expected 2nd entry of type DataLoader, but got {type(test_loader)}!')
      use_test = True
    else:
      raise ValueError(f'Expected tuple of length 2, but got {len(data)}!')
  
  
  #criterion = nn.L1Loss()
  model.train()
  losses = []
  test_losses=[]
  batch_total = len(train_loader)
  best_model=None
  min_loss=np.iinfo(0).max
  for epoch in range(max_epochs):
    #random.seed(42)
    #torch.manual_seed(42)
    #np.random.seed(42)
    running_loss=[]
    test_loss=[]
    for batch_idx, batch in enumerate(train_loader):
      x, y = batch
      if cuda:
        x, y = x.cuda(), y.cuda()
      output = model(x)
      loss = criterion(output, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      running_loss.append(loss.item())
      #rmse += torch.sqrt(criterion(yhat, y))
      #losses.append(loss.item())
      
    if use_test:
      model.eval()
      test_x, test_y =test_loader
      if cuda:
        test_x, test_y = test_x.cuda(), test_y.cuda()
      test_output = model(test_x)
      loss = criterion(test_output, test_y)
      test_loss.append(loss.item())
      #test_mae = criterion(test_output, test_y)
      test_x
      #predictions = scaler.inverse_transform(test_output.cpu().detach().numpy())
      #test_y = scaler.inverse_transform(test_y.cpu().detach().numpy())
      epoch_loss = mean_squared_error(test_y.cpu().detach().numpy(),test_output.cpu().detach().numpy())
      if epoch_loss<min_loss:
        min_loss = epoch_loss
        best_model= model.state_dict()
      test_losses.append(loss.item())
      model.train()
      if epoch%50==0:
        sys.stdout.write(f'\rEpoch: {epoch}/{max_epochs}  Loss: {mean(running_loss):.6f} Test loss: {epoch_loss:.6f}')
    else:
      sys.stdout.write(f'\rEpoch: {epoch}/{max_epochs}  Loss: {running_loss:.6f}' )
    epoch_loss =mean(running_loss)
    losses.append(epoch_loss)
  return (losses, test_losses, best_model)

# In[ ]:




