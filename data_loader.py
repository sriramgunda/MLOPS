import pandas as pd
import numpy as np
import os

def load_data(save_path="data/heart.csv"):
    """
    Downloads the Heart Disease dataset (Cleveland) and saves it locally.
    """
    # Primary Source: UCI Repository URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Fallback Source: Local file
    local_path = "data/heart+disease/processed.cleveland.data"
    
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    
    df = None
    
    # Try downloading from URL first
    print(f"Attempting to download data from {url}...")
    try:
        df = pd.read_csv(url, names=column_names, na_values="?")
        print("Download successful.")
    except Exception as e:
        print(f"Failed to download from URL: {e}")
        
        # Fallback to local file
        print(f"Attempting to load from local file: {local_path}...")
        try:
            if os.path.exists(local_path):
                df = pd.read_csv(local_path, names=column_names, na_values="?")
                print("Local load successful.")
            else:
                raise FileNotFoundError(f"Local file not found at {local_path}")
        except Exception as local_e:
            print(f"Failed to load local data: {local_e}")
            return None
            
    if df is not None:
        print(f"Raw dataset loaded. Shape: {df.shape}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        return df
        
    return df

if __name__ == "__main__":
    load_data()
