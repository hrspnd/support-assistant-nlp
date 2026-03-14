'''
data_pipeline.py
Cleans and filters the dataset to keep only delivery-related intents
and prepares it for intent classification.
'''

import pandas as pd
import os
import re


def clean_text(text):
    """Basic text normalization for customer queries."""
    text = str(text).lower()
    text = text.strip()
    text = re.sub(r"[^\w\s{}]", "", text)  # keep placeholders like {{order number}}
    return text


def main():

    print("Loading raw dataset...")

    df = pd.read_csv("data/raw/bitext_retail_dataset.csv")

    print("Original dataset size:", len(df))

    # Delivery-related intents based on dataset inspection
    delivery_intents = [
        "track_delivery",
        "track_order",
        "delivery_issue",
        "damaged_delivery",
        "delivery_time",
        "missing_item",
        "shipping_costs"
    ]

    # Filter dataset
    df = df[df["intent"].isin(delivery_intents)]

    print("Dataset size after filtering intents:", len(df))

    # Keep only relevant columns
    df = df[["instruction", "intent"]]

    # Rename columns to standard ML format
    df = df.rename(columns={
        "instruction": "text",
        "intent": "label"
    })

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Remove duplicate queries
    df = df.drop_duplicates(subset=["text"])

    print("Dataset size after removing duplicates:", len(df))

    # Create processed folder
    os.makedirs("data/processed", exist_ok=True)

    output_path = "data/processed/intent_dataset.csv"

    # Save processed dataset
    df.to_csv(output_path, index=False)

    print("Processed dataset saved to:", output_path)
    print("Final dataset size:", len(df))


if __name__ == "__main__":
    main()
