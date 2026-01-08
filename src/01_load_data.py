
import pandas as pd
import os

def load_and_summarize(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Basic Info
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    
    # Missing Values
    print("\nMissing Values:\n", df.isnull().sum())
    
    # Statistical Summary
    desc = df.describe()
    print("\nStatistical Summary:\n", desc)
    
    # Save summary report
    os.makedirs('reports', exist_ok=True)
    with open('reports/data_summary.txt', 'w') as f:
        f.write("Data Summary Report\n")
        f.write("===================\n\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write(f"Missing Values:\n{df.isnull().sum()}\n\n")
        f.write(f"Description:\n{desc}\n")
    
    print("\nSummary report saved to reports/data_summary.txt")

    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicates found: {duplicates}")

if __name__ == "__main__":
    load_and_summarize('data/raw/dataset.csv')
