"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to train a neural network model 
    using PyTorch's DataLoader and optimizer while tracking loss and test loss.
Version: 1.0
"""

import os
import sys
import time
import math
import random
import re
import io
import numpy as np
import pandas as pd
import requests
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from datetime import date, timedelta
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
from typing import Union, Tuple, List

# Set Matplotlib style
matplotlib.style.use('seaborn')


def fit(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    data: Union[DataLoader, Tuple[DataLoader, DataLoader]],
    max_epochs: int,
    cuda: bool = True
) -> Tuple[List[float], List[float], OrderedDict]:
    """
    Trains a PyTorch model with the given optimizer and loss criterion.

    Args:
        model (nn.Module): The PyTorch model to train.
        optimizer (optim.Optimizer): The optimizer to update model parameters.
        criterion (nn.Module): The loss function.
        data (Union[DataLoader, Tuple[DataLoader, DataLoader]]): Either a single DataLoader for training 
            or a tuple (train_loader, test_loader).
        max_epochs (int): The maximum number of epochs for training.
        cuda (bool, optional): Whether to use GPU. Defaults to True.

    Returns:
        Tuple[List[float], List[float], OrderedDict]: A tuple containing:
            - training losses per epoch
            - test losses per epoch (if applicable)
            - the best model's state dictionary (lowest test loss)
    """
    use_test = False

    if isinstance(data, DataLoader):
        train_loader = data
    elif isinstance(data, tuple):
        if len(data) == 2:
            train_loader, test_loader = data
            if not isinstance(train_loader, DataLoader):
                raise TypeError(f'Expected 1st entry of type DataLoader, but got {type(train_loader)}!')
            if not isinstance(test_loader, DataLoader):
                raise TypeError(f'Expected 2nd entry of type DataLoader, but got {type(test_loader)}!')
            use_test = True
        else:
            raise ValueError(f'Expected tuple of length 2, but got {len(data)}!')

    model.train()
    losses: List[float] = []
    test_losses: List[float] = []
    min_loss = float('inf')
    best_model: OrderedDict = None  # Stores the best model parameters

    for epoch in range(max_epochs):
        running_loss: List[float] = []
        test_loss: List[float] = []

        for batch_idx, batch in enumerate(train_loader, start=1):
            x, y = batch
            if cuda and torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        # Test evaluation (if test_loader is provided)
        if use_test:
            model.eval()
            test_x, test_y = next(iter(test_loader))  # Get a single batch for testing

            if cuda and torch.cuda.is_available():
                test_x, test_y = test_x.cuda(), test_y.cuda()

            test_output = model(test_x)
            loss = criterion(test_output, test_y)
            test_loss.append(loss.item())

            epoch_loss = mean_squared_error(test_y.cpu().detach().numpy(), test_output.cpu().detach().numpy())

            if epoch_loss < min_loss:
                min_loss = epoch_loss
                best_model = model.state_dict()

            test_losses.append(loss.item())
            model.train()

            if epoch % 50 == 0:
                sys.stdout.write(f'\rEpoch: {epoch}/{max_epochs}  Loss: {mean(running_loss):.6f}  Test loss: {epoch_loss:.6f}\n')
        else:
            sys.stdout.write(f'\rEpoch: {epoch}/{max_epochs}  Loss: {mean(running_loss):.6f}\n')

        losses.append(mean(running_loss))

    return losses, test_losses, best_model

