
import pandas as pd
import numpy as np
import os

def clean_data(input_file, output_file):
    print(f"Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)
    
    initial_shape = df.shape
    print(f"Initial shape: {initial_shape}")
    
    # 1. Handle Missing Values
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Found missing values:\n{missing[missing > 0]}")
        # For this dataset, we might drop rows or fill numeric with median
        # For simplicity in this pipeline, we'll fill with median for numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill categorical with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        print("Missing values handled.")
    else:
        print("No missing values found.")

    # 2. Data Types & Cleaning
    # Round columns with excessive precision (like Total_Screen_Time)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].round(2)
    print("Floats rounded to 2 decimal places.")

    # 3. Save Processed Data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Clean data saved to {output_file}")
    
    print(f"Final shape: {df.shape}")
    
if __name__ == "__main__":
    clean_data('data/raw/dataset.csv', 'data/processed/clean_dataset.csv')
