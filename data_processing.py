import pandas as pd
import numpy as np
from scipy import stats

def load_dataset(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def remove_duplicates(df):
    """Remove duplicate rows from the dataset."""
    return df.drop_duplicates()

def handle_missing_values(df, method="mean"):
    """Handle missing values using a specified method."""
    if method == "mean":
        return df.fillna(df.mean())
    elif method == "median":
        return df.fillna(df.median())
    elif method == "mode":
        return df.fillna(df.mode().iloc[0])
    elif method == "drop":
        return df.dropna()
    else:
        raise ValueError("Invalid method for handling missing values.")

def detect_outliers(df, threshold=3):
    """Detect outliers using the Z-score method."""
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    return (z_scores > threshold).any(axis=1)

def handle_outliers(df, method="remove", threshold=3):
    """Handle outliers based on user-defined method."""
    outliers = detect_outliers(df, threshold)
    if method == "remove":
        return df[~outliers]
    elif method == "replace_mean":
        df.loc[outliers, df.select_dtypes(include=[np.number]).columns] = df.mean()
    elif method == "replace_median":
        df.loc[outliers, df.select_dtypes(include=[np.number]).columns] = df.median()
    return df

def normalize_data(df):
    """Normalize numerical features using Min-Max scaling."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    return df

def main():
    file_path = input("Enter dataset file path: ")
    df = load_dataset(file_path)
    print("Dataset loaded successfully.")
    
    # Remove duplicates
    df = remove_duplicates(df)
    print("Duplicates removed.")
    
    # Handle missing values
    method = input("Choose missing value handling method (mean/median/mode/drop): ")
    df = handle_missing_values(df, method)
    print("Missing values handled.")
    
    # Handle outliers
    threshold = int(input("Enter outlier threshold (e.g., 3 for 3 standard deviations): "))
    outlier_method = input("Choose outlier handling method (remove/replace_mean/replace_median): ")
    df = handle_outliers(df, outlier_method, threshold)
    print("Outliers handled.")
    
    # Normalize data
    df = normalize_data(df)
    print("Data normalized.")
    
    # Save cleaned data
    df.to_csv("cleaned_data.csv", index=False)
    print("Cleaned dataset saved as cleaned_data.csv")

if __name__ == "__main__":
    main()
