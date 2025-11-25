"""
Data Preprocessing Step — SageMaker-compatible
Performs basic data cleaning and encoding.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.encoder = LabelEncoder()

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing numeric values with mean and categorical with mode."""
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown", inplace=True)
        print(" Missing values handled.")
        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label encode all categorical columns."""
        for col in df.select_dtypes(include=["object", "category"]).columns:
            df[col] = self.encoder.fit_transform(df[col].astype(str))
        print(" Categorical features encoded.")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full preprocessing pipeline."""
        df = self.handle_missing(df)
        df = self.encode_categorical(df)
        print(f" Preprocessing complete. Final shape: {df.shape}")
        return df

def main():
    import sys
    
    # SageMaker input/output paths
    input_dir = "/opt/ml/processing/input"
    output_dir = "/opt/ml/processing/output"
    
    # Create output subdirectories for train and test
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"📁 Created output directories: {train_dir}, {test_dir}")

    # Find input CSV file (SageMaker may extract to subdirectories)
    input_file = None
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                input_file = os.path.join(root, file)
                print(f"📁 Found CSV file: {input_file}")
                break
        if input_file:
            break
    
    if not input_file:
        print(f"❌ ERROR: No CSV file found in {input_dir}")
        print(f"Contents of {input_dir}:")
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path):
                print(f"  - {item}/ (directory)")
            else:
                print(f"  - {item}")
        sys.exit(1)
    
    # Read input CSV
    df = pd.read_csv(input_file)
    print(f"✅ Loaded input data: {df.shape}")

    # Identify target column for verification
    if 'target' in df.columns:
        target_col = 'target'
    elif 'Target' in df.columns:
        target_col = 'Target'
    elif 'Heart Disease' in df.columns:
        target_col = 'Heart Disease'
    elif 'heart_disease' in df.columns:
        target_col = 'heart_disease'
    else:
        target_col = df.columns[-1]  # Use last column as target
    print(f"📌 Target column identified: {target_col}")

    # Preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(df)
    print(f"✅ Preprocessing complete: {df.shape}")

    # Split into train and test (80/20 split)
    from sklearn.model_selection import train_test_split
    
    # Verify target column still exists after preprocessing
    if target_col not in df.columns:
        print(f"❌ ERROR: Target column '{target_col}' not found after preprocessing!")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Check target values after preprocessing (should be 0/1 for binary classification)
    target_values = df[target_col].unique()
    print(f"📊 Target values after preprocessing: {sorted(target_values)}")
    print(f"📊 Target data type: {df[target_col].dtype}")
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"✅ Split data - Train: {train_df.shape}, Test: {test_df.shape}")

    # Save train and test data to SageMaker output folders
    train_file = os.path.join(train_dir, "train.csv")
    test_file = os.path.join(test_dir, "test.csv")
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f" Train data saved at {train_file}")
    print(f" Test data saved at {test_file}")
    
    # Verify files were written
    if os.path.exists(train_file) and os.path.exists(test_file):
        train_size = os.path.getsize(train_file)
        test_size = os.path.getsize(test_file)
        print(f" Verified: Train file ({train_size} bytes), Test file ({test_size} bytes)")
    else:
        print(f"❌ ERROR: Output files were not created!")
        sys.exit(1)
    
    print("✅ Preprocessing step completed successfully.")

if __name__ == "__main__":
    main()
