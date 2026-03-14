# Fine-tunes a BERT model for intent classification using the processed dataset.

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)


def compute_metrics(eval_pred):
    """Compute evaluation metrics for validation."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def main():

    print("Checking device...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading processed dataset...")

    df = pd.read_csv("data/processed/intent_dataset.csv")

    # Convert labels to numeric IDs
    label_list = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(label_list)}
    df["label"] = df["label"].map(label2id)

    print("Labels:", label2id)

    # Train/validation split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_list)
    )

    model.to(device)

    # Training configuration -- LOWER Batch sizes and only 1 epoch | FOR WEEK 2 ONLY
    training_args = TrainingArguments(
        output_dir="models/bert_intent_classifier",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="experiments/logs",
        load_best_model_at_end=True,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("Starting training...")

    trainer.train()

    print("Saving model...")

    trainer.save_model("models/bert_intent_classifier")

    # Save tokenizer so the model can be reused later
    tokenizer.save_pretrained("models/bert_intent_classifier")

    print("Training complete.")


if __name__ == "__main__":
    main()
