'''
This script downloads the Bitext customer support dataset from Hugging Face
and saves it locally as a CSV file inside the data/raw directory.
'''
from datasets import load_dataset
import pandas as pd
import os


def main():

    print("Downloading dataset from Hugging Face...")

    dataset = load_dataset("bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset")

    # Convert the training split to a pandas DataFrame
    df = dataset["train"].to_pandas()

    # Create raw data folder if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)

    output_path = "data/raw/bitext_retail_dataset.csv"

    # Save dataset locally
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print(f"Total rows: {len(df)}")
    print("Columns:", df.columns)


if __name__ == "__main__":
    main()
