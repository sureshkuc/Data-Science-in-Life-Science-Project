"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Description:
    This module provides functions for handling COVID-19 data, including fetching data,
    visualizing trends, and uploading datasets to GitHub.
Version: 1.0
"""

import os
import io
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
from github import Github
from typing import Optional

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Use seaborn style for better visuals
plt.style.use('seaborn')

def fetch_covid_data(url: str) -> pd.DataFrame:
    """
    Fetches COVID-19 data from the given URL and loads it into a Pandas DataFrame.
    
    Args:
        url (str): The URL containing the COVID-19 dataset in CSV format.
    
    Returns:
        pd.DataFrame: A DataFrame containing the COVID-19 data.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP failures
        df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
        return df
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if request fails

def visualize_covid_trends(df: pd.DataFrame) -> None:
    """
    Generates a line plot of cumulative confirmed cases for select Indian states.

    Args:
        df (pd.DataFrame): DataFrame containing COVID-19 data.
    """
    selected_states = ['Karnataka', 'Maharashtra', 'Uttar Pradesh', 'Kerala', 'Tamil Nadu']
    fig = plt.figure(figsize=(15, 8))
    
    with plt.xkcd():
        ax = sns.lineplot(
            data=df[df["State"].isin(selected_states)],
            x="Date",
            y="Confirmed",
            hue="State"
        )
        ax.set_title("Cumulative Confirmed Cases by States", size=20)
    
    plt.show()

def github_upload(folder_name: str, file_name: str, file_data: str) -> None:
    """
    Uploads a file to a GitHub repository.

    Args:
        folder_name (str): The folder name where the file should be stored in GitHub.
        file_name (str): The name of the file.
        file_data (str): The content of the file.

    Notes:
        - Ensure you have set a GitHub token as an environment variable `GITHUB_TOKEN`
          to authenticate API requests securely.
    """
    github_token = os.getenv("GITHUB_TOKEN")  # Fetch API token securely
    if not github_token:
        print("Error: GitHub token not found. Set 'GITHUB_TOKEN' as an environment variable.")
        return
    
    try:
        g = Github(github_token)
        repo_name = "Data-Science-in-Life-Science-Project"
        repo = g.get_user().get_repo(repo_name)

        all_files = []
        contents = repo.get_contents("")
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                contents.extend(repo.get_contents(file_content.path))
            else:
                all_files.append(file_content.path)

        git_file_path = f"{folder_name}/{file_name}"

        if git_file_path in all_files:
            contents = repo.get_contents(git_file_path)
            repo.update_file(contents.path, "Updating file", file_data, contents.sha, branch="main")
            print(f"{git_file_path} UPDATED")
        else:
            repo.create_file(git_file_path, "Creating new file", file_data, branch="main")
            print(f"{git_file_path} CREATED")
    
    except Exception as e:
        print(f"GitHub upload error: {e}")

def preprocess_state_data(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """
    Preprocesses COVID-19 data for a specific state, adding new columns for daily changes.

    Args:
        df (pd.DataFrame): The full COVID-19 dataset.
        state (str): The state for which the data needs to be processed.

    Returns:
        pd.DataFrame: Processed DataFrame with new columns for daily confirmed, deaths, and recovered cases.
    """
    state_df = df[df["State"] == state].copy()
    
    state_df["New_Confirmed"] = np.nan
    state_df["New_Deaths"] = np.nan
    state_df["New_Recovered"] = np.nan

    # Compute daily changes
    state_df.loc[1:, "New_Confirmed"] = state_df["Confirmed"].values[1:] - state_df["Confirmed"].values[:-1]
    state_df.loc[1:, "New_Deaths"] = state_df["Deceased"].values[1:] - state_df["Deceased"].values[:-1]
    state_df.loc[1:, "New_Recovered"] = state_df["Recovered"].values[1:] - state_df["Recovered"].values[:-1]

    return state_df

def main() -> None:
    """
    Main function to execute the workflow: fetch data, process it, visualize trends, 
    and optionally upload processed data to GitHub.
    """
    # Fetch COVID-19 data
    url = "https://api.covid19india.org/csv/latest/states.csv"
    df = fetch_covid_data(url)
    
    if df.empty:
        print("No data available. Exiting.")
        return

    # Visualize COVID trends for selected states
    visualize_covid_trends(df)

    # Process and upload data for selected states
    states_to_upload = ["Karnataka", "Maharashtra", "Uttar Pradesh", "Kerala", "Tamil Nadu"]
    
    for state in states_to_upload:
        processed_df = preprocess_state_data(df, state)
        
        # Convert processed data to CSV string
        csv_content = processed_df.to_csv(index=False)

        # Upload the data to GitHub
        github_upload(
            folder_name="Indian-States-Covid19-Datasets",
            file_name=f"{state.replace(' ', '-')}.csv",
            file_data=csv_content
        )

if __name__ == "__main__":
    main()

