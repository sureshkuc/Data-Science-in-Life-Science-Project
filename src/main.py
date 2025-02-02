"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions and classes to train and evaluate CNN, GRU, LSTM, and MLP models
    on COVID-19 datasets from selected Indian states. The models are trained using different hyperparameters
    and evaluated based on MAE, RMSE, and R2 Score.
Version: 1.0
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
from model import CNNModel, GRUNet, LSTM, MLP
from evaluation import evaluate_model
from config import SHORTLISTED_STATES, TIME_STEPS, NUMBER_OF_FEATURES, OUTPUT_DIM, SEED
from data_preparation import data_preparation
from model_fit_code import fit

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="outputs/training.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)
logger = logging.getLogger()

def set_random_seed(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info("Random seed set to %d", seed)

def train_and_evaluate(model_class, model_name: str) -> pd.DataFrame:
    """
    Trains and evaluates the specified model on COVID-19 datasets.
    
    Args:
        model_class: The model class (CNNModel, GRUNet, LSTM, MLP).
        model_name (str): The name of the model for logging.
    
    Returns:
        pd.DataFrame: A DataFrame containing evaluation results for different states and hyperparameters.
    """
    results = []
    set_random_seed(SEED)
    
    for state in SHORTLISTED_STATES:
        try:
            df = pd.read_csv(
                f"https://raw.githubusercontent.com/sureshkuc/Data-Science-in-Life-Science-Project/main/Indian-States-Covid19-Datasets/{state}.csv",
                parse_dates=["Date"]
            ).drop(columns=["Unnamed: 0"], errors='ignore')
            df = df[df["Date"] > "2020-03-10"].set_index("Date")
            df = df[['Confirmed', 'Recovered', 'Deceased', 'New_Confirmerd', 'New_Deaths', 'New_Recovered']]
        except Exception as e:
            logger.error("Error loading data for state %s: %s", state, e)
            continue
        
        for n_f in NUMBER_OF_FEATURES:
            for t_s in TIME_STEPS:
                try:
                    train_loader, test_loader, scaler = data_preparation(df, scaling_range=(0, 1), time_step=t_s, number_feature=n_f, response_variable_index=0, data_split_ratio=0.8)
                except Exception as e:
                    logger.error("Data preparation failed for state %s, n_f=%d, t_s=%d: %s", state, n_f, t_s, e)
                    continue
                
                for n_layers in range(1, 4):
                    for hidden_param in [1, 5, 8, 16, 32]:
                        try:
                            max_epochs = 25 if model_name == "GRU" else 100
                            model = model_class(n_f, hidden_param, OUTPUT_DIM, n_layers, t_s) if model_name in ["LSTM", "GRU"] else model_class(input_dim=n_f * t_s, layers=n_layers, output_dim=OUTPUT_DIM)
                            optimizer = optim.Adam(model.parameters(), lr=1e-3) if model_name == "GRU" else optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

                            train_losses, test_losses, best_model = fit(model, optimizer, nn.L1Loss(), (train_loader, test_loader), max_epochs=max_epochs, cuda=False)
                            model.load_state_dict(best_model)
                            model.eval()

                            mae, rmse, r2s = evaluate_model(model, test_loader)
                            results.append([state, n_f, t_s, n_layers, hidden_param, mae, rmse, r2s])
                            logger.info("%s | %s | n_f=%d, t_s=%d, n_layers=%d, param=%d | MAE=%.4f, RMSE=%.4f, R2=%.4f", model_name, state, n_f, t_s, n_layers, hidden_param, mae, rmse, r2s)
                        except Exception as e:
                            logger.error("Model training failed for %s, state=%s, n_f=%d, t_s=%d, n_layers=%d, param=%d: %s", model_name, state, n_f, t_s, n_layers, hidden_param, e)
                            continue
    
    return pd.DataFrame(results, columns=['State', 'Number_feature', 'Time_Step', 'Number_Layers', 'Parameter', 'MAE', 'RMSE', 'R2_Score'])

def main() -> None:
    """Main function to run experiments for different models."""
    try:
        logger.info("Starting model training and evaluation...")
        df_cnn = train_and_evaluate(CNNModel, "CNN")
        df_gru = train_and_evaluate(GRUNet, "GRU")
        df_lstm = train_and_evaluate(LSTM, "LSTM")
        df_mlp = train_and_evaluate(MLP, "MLP")

        logger.info("Training completed successfully.")
        print("\nCNN Model Results:\n", df_cnn.head())
        print("\nGRU Model Results:\n", df_gru.head())
        print("\nLSTM Model Results:\n", df_lstm.head())
        print("\nMLP Model Results:\n", df_mlp.head())
    except Exception as e:
        logger.critical("Unexpected error in main function: %s", e, exc_info=True)

if __name__ == "__main__":
    main()


