"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions for training and evaluating a GRU-based model 
    using PyTorch. It includes functions for model fitting and performance evaluation 
    using metrics like MAE, RMSE, and R2 Score.
Version: 1.0
"""

import torch
import torch.optim as optim
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, List, Dict
from model import GRUNet  # Ensure `GRUNet` is defined in `model.py`


def evaluate_model(test_y: np.ndarray, predictions: np.ndarray) -> Tuple[float, float, float]:
    """
    Evaluates the model's performance using Mean Absolute Error (MAE), 
    Root Mean Squared Error (RMSE), and R² score.
    
    Args:
        test_y (np.ndarray): Ground truth values.
        predictions (np.ndarray): Model predictions.

    Returns:
        Tuple[float, float, float]: MAE, RMSE, and R² score.
    """
    mae = mean_absolute_error(test_y, predictions)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    r2s = r2_score(test_y, predictions)
    return mae, rmse, r2s


def fit(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    criterion: torch.nn.Module,
    data: Tuple[torch.utils.data.DataLoader, Tuple[torch.Tensor, torch.Tensor]],
    max_epochs: int,
    cuda: bool = False
) -> Tuple[List[float], List[float], Dict]:
    """
    Trains a given PyTorch model using the provided optimizer and loss function.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (optim.Optimizer): Optimizer for training.
        criterion (torch.nn.Module): Loss function.
        data (Tuple): A tuple containing the training DataLoader and test dataset.
        max_epochs (int): The maximum number of training epochs.
        cuda (bool, optional): Whether to use GPU acceleration. Defaults to False.

    Returns:
        Tuple[List[float], List[float], Dict]: Training losses, test losses, and the best model state dictionary.
    """
    train_loader, test_loader = data
    model.train()
    losses = []
    test_losses = []
    best_model = None
    min_loss = np.inf  # Use np.inf instead of np.iinfo(0).max

    for epoch in range(max_epochs):
        running_loss = []
        test_loss = []
        
        # Training phase
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            if cuda:
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

        # Evaluation phase
        model.eval()
        test_x, test_y = test_loader
        if cuda:
            test_x, test_y = test_x.cuda(), test_y.cuda()

        with torch.no_grad():
            test_output = model(test_x)
            loss = criterion(test_output, test_y)
            test_loss.append(loss.item())

            epoch_loss = mean_squared_error(
                test_y.cpu().numpy(), test_output.cpu().numpy()
            )
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                best_model = model.state_dict()

        test_losses.append(loss.item())
        model.train()

    return losses, test_losses, best_model


def evaluate_trained_model(model: torch.nn.Module, test_loader: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, float, float]:
    """
    Evaluates a trained PyTorch model on a test dataset.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (Tuple[torch.Tensor, torch.Tensor]): Test dataset (features and labels).

    Returns:
        Tuple[float, float, float]: MAE, RMSE, and R² score.
    """
    model.eval()
    test_x, test_y = test_loader

    with torch.no_grad():
        predictions = model(test_x)

    test_y_np = test_y.cpu().numpy()
    predictions_np = predictions.cpu().numpy()

    return evaluate_model(test_y_np, predictions_np)

