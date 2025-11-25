"""
Data Fetching Step for SageMaker Pipeline
Reads input CSV or downloads from a URL and saves to /opt/ml/processing/output
"""

import os
import pandas as pd
import requests

# SageMaker Processing input/output directories
INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"

class DataFetcher:
    def __init__(self, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
        os.makedirs(output_dir, exist_ok=True)
        self.input_dir = input_dir
        self.output_dir = output_dir

    def fetch_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        df = pd.read_csv(file_path)
        print(f"✅ Loaded CSV: {file_path} — shape: {df.shape}")
        return df

    def fetch_from_url(self, url: str, save_name: str = "downloaded.csv") -> pd.DataFrame:
        """Download a CSV from a URL and load it."""
        response = requests.get(url)
        response.raise_for_status()
        file_path = os.path.join(self.output_dir, save_name)
        with open(file_path, "wb") as f:
            f.write(response.content)
        df = pd.read_csv(file_path)
        print(f"✅ Downloaded from {url} — saved to {file_path} — shape: {df.shape}")
        return df

    def save_data(self, df: pd.DataFrame, filename: str = "fetched_data.csv"):
        """Save DataFrame to SageMaker output folder."""
        save_path = os.path.join(self.output_dir, filename)
        df.to_csv(save_path, index=False)
        print(f"💾 Data saved at {save_path}")

def main():
    import sys
    
    fetcher = DataFetcher()
    
    # SageMaker may copy files to subdirectories, so search for the CSV file
    input_file = None
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith('.csv'):
                input_file = os.path.join(root, file)
                print(f"📁 Found CSV file: {input_file}")
                break
        if input_file:
            break
    
    if not input_file:
        print(f"❌ ERROR: No CSV file found in {INPUT_DIR}")
        print(f"Contents of {INPUT_DIR}:")
        for item in os.listdir(INPUT_DIR):
            print(f"  - {item}")
        sys.exit(1)
    
    # Load the CSV file
    df = fetcher.fetch_from_csv(input_file)
    print(f"✅ Loaded data with shape: {df.shape}")

    # Save fetched data to output folder for SageMaker to upload to S3
    output_file = os.path.join(OUTPUT_DIR, "fetched_data.csv")
    df.to_csv(output_file, index=False)
    print(f"💾 Data saved at {output_file}")
    
    # Verify file was written
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"✅ Verified: Output file exists ({file_size} bytes)")
    else:
        print(f"❌ ERROR: Output file was not created!")
        sys.exit(1)

    print("✅ Data fetch step completed successfully.")

if __name__ == "__main__":
    main()
