# Fine-tunes a BERT model for intent classification using the processed dataset.

import pandas as pd
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "intent_dataset.csv")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    _, _, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}


class CleanLogCallback(TrainerCallback):
    """Prints only epoch loss, accuracy, and F1 — suppresses everything else."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if "eval_accuracy" in logs:
            epoch = int(state.epoch)
            loss  = logs.get("loss", logs.get("eval_loss", 0.0))
            acc   = logs["eval_accuracy"]
            f1    = logs["eval_f1"]
            print(f"\n\nEpoch {epoch}, Loss: {loss:.4f}")
            print(f"\nBERT Results:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1 Score: {f1:.4f}\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    df = pd.read_csv(DATA_PATH)

    label_list = sorted(df["label"].unique())
    label2id   = {label: i for i, label in enumerate(label_list)}
    df["label"] = df["label"].map(label2id)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset   = Dataset.from_pandas(val_df)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=32
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset   = val_dataset.map(tokenize, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch",   columns=["input_ids", "attention_mask", "label"])

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_list)
    )
    model.to(device)

    training_args = TrainingArguments(
        output_dir="models/bert_intent_classifier",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="experiments/logs",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        disable_tqdm=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[CleanLogCallback()],
    )

    print("Starting training...\n")
    trainer.train()

    # ── Extract F1 per epoch ───────────────────────────────────────────────────
    history = trainer.state.log_history
    bert_f1 = [log["eval_f1"] for log in history if "eval_f1" in log]

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/bert_f1_history.json", "w") as f:
        json.dump(bert_f1, f)

    # ── Learning curve ─────────────────────────────────────────────────────────
    margin = 0.01
    all_vals = bert_f1

    plt.figure()
    plt.plot(range(1, len(bert_f1) + 1), bert_f1, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("BERT Learning Curve (F1 Score)")
    plt.ylim(min(all_vals) - margin, min(max(all_vals) + margin, 1.0))
    plt.xticks(range(1, len(bert_f1) + 1))
    plt.savefig("experiments/f1_learning_curve_bert.png")
    plt.show()

    # ── Confusion matrix ───────────────────────────────────────────────────────
    print("Generating confusion matrix...")

    preds_output = trainer.predict(val_dataset)
    preds        = preds_output.predictions.argmax(axis=-1)
    true         = preds_output.label_ids

    cm  = confusion_matrix(true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)

    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title("BERT Confusion Matrix")
    plt.tight_layout()
    plt.savefig("experiments/confusion_matrix_bert.png")
    plt.show()

    # ── Save model ─────────────────────────────────────────────────────────────
    trainer.save_model("models/bert_intent_classifier")
    tokenizer.save_pretrained("models/bert_intent_classifier")

    print("\nTraining complete.")
    print("Label mapping:", label2id)


if __name__ == "__main__":
    main()
