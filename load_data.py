import os
import pandas as pd

def load_dataset(file_path):
    """Load dataset and perform initial checks."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERROR: File not found at {file_path}. Please check the path.")
    
    df = pd.read_csv(file_path)
    print("\n✅ Dataset loaded successfully!")
    print("\n🔍 Basic Info:")
    print(df.info())
    
    print("\n📊 First 5 rows of the dataset:")
    print(df.head())
    
    print("\n🔹 Checking for missing values:")
    print(df.isnull().sum())
    
    print("\n🔹 Checking for duplicates:")
    print("Duplicates found:", df.duplicated().sum())
    
    return df

def main():
    file_path = input("Enter dataset file path: ")
    df = load_dataset(file_path)

if __name__ == "__main__":
    main()
