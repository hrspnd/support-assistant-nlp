# inspect_dataset.py
# This script downloads the Bitext retail ecommerce dataset
# and prints useful information about its structure and intents.

from datasets import load_dataset


def main():

    print("Loading dataset...")

    dataset = load_dataset("bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset")

    # Convert to pandas dataframe
    df = dataset["train"].to_pandas()

    print("\nTotal rows:", len(df))

    print("\nColumns in dataset:")
    print(df.columns)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nIntent distribution:")
    print(df["intent"].value_counts())

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    print("\nExample queries:")
    print(df["instruction"].sample(10))


if __name__ == "__main__":
    main()
