"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and configurations for setting up and training a neural network
    using PyTorch. It includes data preprocessing, setting random seeds for reproducibility, and
    defining key hyperparameters for model training.
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available")

def set_seed(seed: int = 42) -> None:
    """
    Set the seed for random number generation to ensure reproducibility.

    Args:
        seed (int): The seed value to use for numpy, torch, and random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Configuration settings
SHORTLISTED_STATES = ['Karnataka', 'Maharashtra', 'Uttar-Pradesh', 'Kerala', 'Tamil-Nadu']
TIME_STEPS = [5, 7, 15, 30]
NUMBER_OF_FEATURES = [1, 2, 3, 4, 5, 6]
OUTPUT_DIM = 1
SEED = 42

# Set seed for reproducibility
set_seed(SEED)

if __name__ == "__main__":
    print("Configuration and utility functions loaded successfully.")
