"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions for preparing and processing datasets 
    for deep learning models using PyTorch. It includes functions for 
    dataset creation, normalization, and train-test splitting.
Version: 1.0
"""

import os
import sys
import time
import math
import random
import re
import io
import imageio
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import date, timedelta
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statistics import mean
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Union

# Set Matplotlib style
matplotlib.style.use('seaborn')


def create_dataset(
    dataset: np.ndarray,
    look_back: int = 1,
    response_variable_index: int = 0,
    number_feature: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts an array of values into a dataset matrix suitable for time-series forecasting.

    Args:
        dataset (np.ndarray): The input dataset as a NumPy array.
        look_back (int): Number of time steps to look back.
        response_variable_index (int): Index of the target variable in the dataset.
        number_feature (int): Number of features in the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix (X) and target vector (Y).
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :number_feature]
        dataX.append(a)
        dataY.append(dataset[i + look_back, response_variable_index])
    
    return np.array(dataX), np.array(dataY)


def data_preparation(
    df: pd.DataFrame,
    scaling_range: Tuple[int, int] = (0, 1),
    time_step: int = 5,
    number_feature: int = 6,
    response_variable_index: int = 3,
    data_split_ratio: float = 0.8,
    shuffle: bool = True
) -> Tuple[DataLoader, Tuple[torch.Tensor, torch.Tensor], MinMaxScaler]:
    """
    Prepares the dataset for deep learning by normalizing, splitting into train/test sets,
    and creating PyTorch DataLoaders.

    Args:
        df (pd.DataFrame): Input DataFrame containing the dataset.
        scaling_range (Tuple[int, int]): Min-max scaling range. Default is (0,1).
        time_step (int): Number of time steps for creating sequences.
        number_feature (int): Number of features to consider.
        response_variable_index (int): Index of the target variable.
        data_split_ratio (float): Ratio to split train and test data.
        shuffle (bool): Whether to shuffle the dataset during training.

    Returns:
        Tuple[DataLoader, Tuple[torch.Tensor, torch.Tensor], MinMaxScaler]: 
        - DataLoader for training.
        - Tuple containing test dataset (features, targets).
        - Scaler object for inverse transformation.
    """
    df = df.astype('float32')

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=scaling_range)
    dataset = scaler.fit_transform(df)

    # Create dataset sequences
    X, Y = create_dataset(dataset, time_step, response_variable_index, number_feature)

    # Split into train and test sets
    train_size = int(len(dataset) * data_split_ratio)
    trainX, testX = X[:train_size], X[train_size:]
    trainY, testY = Y[:train_size], Y[train_size:]

    print(f"Training data shape: {trainX.shape}")
    
    # Reshape input for PyTorch
    if number_feature == 1:
        trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
        testX = testX.reshape(testX.shape[0], testX.shape[1], 1)

    print(f"Train shape: {trainX.shape}, {trainY.shape}")

    # Convert to PyTorch tensors
    inputs_train = torch.from_numpy(trainX)
    targets_train = torch.from_numpy(trainY.reshape(-1, 1))

    # Create PyTorch dataset
    train_ds = TensorDataset(inputs_train, targets_train)
    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

    # Prepare test dataset
    inputs_test = torch.from_numpy(testX)
    targets_test = torch.from_numpy(testY.reshape(-1, 1))

    test_ds = (inputs_test, targets_test)

    return train_loader, test_ds, scaler


if __name__ == "__main__":
    # Example usage
    print("This module is intended for data processing and does not run standalone.")

