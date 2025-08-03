# scripts/get_data.py
import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path

def get_raw_data():
    """Loads the Iris dataset and saves it to the raw data directory."""
    print("Fetching raw data...")
    # Load dataset
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]

    # Define save path and create directory if it doesn't exist
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "iris.csv"

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Raw data saved to {output_path}")

if __name__ == "__main__":
    get_raw_data()