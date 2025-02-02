"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides neural network models including CNN, GRU, LSTM, and MLP 
    for sequence and feature-based data processing using PyTorch.
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from config import cuda, SEED

class Flatten(nn.Module):
    """A helper class to flatten the output tensor to 2D."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flattens the input tensor except the batch dimension."""
        return x.view(x.size(0), -1)

class CNNModel(nn.Module):
    """A Convolutional Neural Network (CNN) for sequence data processing."""
    def __init__(self, time_step: int, n_layers: int, vector_length: int, kernel_size: int) -> None:
        """Initializes the CNN model.
        
        Args:
            time_step (int): Number of time steps.
            n_layers (int): Number of CNN layers.
            vector_length (int): Length of input vectors.
            kernel_size (int): Size of the convolution kernel.
        """
        super().__init__()
        self.time_step = time_step
        self.n_layers = n_layers
        in_channels = 1
        out_channels = 16
        layers = []
        
        for l in range(self.n_layers):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding="same"))
            layers.append(nn.Tanh() if l % 2 == 0 else nn.ELU(inplace=True))
            in_channels = out_channels
            out_channels *= 2
        
        layers.append(Flatten())
        out = int(vector_length * (out_channels / 2))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of CNNModel.
        
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        b, features, look_back = x.shape
        x = x.view(b, 1, features * look_back)
        y = self.body(x)
        return self.head(y.view(len(y), -1))

class GRUNet(nn.Module):
    """A Gated Recurrent Unit (GRU) network for sequence modeling."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int) -> None:
        """Initializes the GRU network.
        
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer size.
            output_dim (int): Output dimension.
            n_layers (int): Number of GRU layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of GRUNet.
        
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        h = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.gru(x, h)
        return self.fc(self.relu(out[:, -1]))

class LSTM(nn.Module):
    """A Long Short-Term Memory (LSTM) network for sequence modeling."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, seq_length: int) -> None:
        """Initializes the LSTM network.
        
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer size.
            output_dim (int): Output dimension.
            num_layers (int): Number of LSTM layers.
            seq_length (int): Sequence length.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.relu = nn.ELU()
        self.fc = nn.Linear(hidden_dim * seq_length, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of LSTM.
        
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        x = out.contiguous().view(batch_size, -1)
        return self.fc(self.relu(x))

class MLP(nn.Module):
    """A Multi-Layer Perceptron (MLP) model for feature-based data processing."""
    def __init__(self, input_dim: int, layers: int, output_dim: int) -> None:
        """Initializes the MLP model.
        
        Args:
            input_dim (int): Input feature dimension.
            layers (int): Number of hidden layers.
            output_dim (int): Output dimension.
        """
        super().__init__()
        self.n_layers = layers
        in_features = input_dim
        out_features = 16
        layer_list = []

        for l in range(self.n_layers):
            if l == self.n_layers - 1:
                layer_list.append(nn.Linear(in_features, output_dim))
            else:
                layer_list.append(nn.Linear(in_features, out_features))
                layer_list.append(nn.Tanh() if l % 2 == 0 else nn.ELU(inplace=True))
                in_features = out_features
                out_features = max(out_features // 2, 4)

        self.body = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP.
        
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        b, n_steps, features = x.shape
        x = x.view(b, n_steps * features)
        return self.body(x)

