'''
This script downloads the Bitext customer support dataset from Hugging Face
and saves it locally as a CSV file inside the data/raw directory.
'''

from datasets import load_dataset
import pandas as pd
import os


def main():

    # Inform the user that the dataset download is starting
    print("Downloading Bitext customer support dataset...")

    # Load the dataset from Hugging Face
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

    # Convert the training split into a pandas DataFrame
    df = dataset["train"].to_pandas()

    # Create the data/raw folder if it does not already exist
    os.makedirs("data/raw", exist_ok=True)

    # Define the path where the dataset will be saved
    output_path = "data/raw/bitext_customer_support.csv"

    # Save the dataset as a CSV file
    df.to_csv(output_path, index=False)

    # Print confirmation and dataset size
    print(f"Dataset saved to {output_path}")
    print(f"Total rows: {len(df)}")


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
