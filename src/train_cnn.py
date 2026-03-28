# train_cnn.py

import pandas as pd
import torch
import torch.nn as nn
import os
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "intent_dataset.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Encode labels
label_list = sorted(df["label"].unique())
label2id   = {label: i for i, label in enumerate(label_list)}
df["label"] = df["label"].map(label2id)

# Train/val split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


# Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=50,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "label":     torch.tensor(self.labels[idx])
        }


# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
val_dataset   = TextDataset(val_df["text"].tolist(),   val_df["label"].tolist(),   tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32)


# Text-CNN Model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 100, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, 100, kernel_size=4)
        self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=5)
        self.fc    = nn.Linear(300, num_classes)

    def forward(self, x):
        x  = self.embedding(x).permute(0, 2, 1)
        x1 = torch.max(torch.relu(self.conv1(x)), dim=2)[0]
        x2 = torch.max(torch.relu(self.conv2(x)), dim=2)[0]
        x3 = torch.max(torch.relu(self.conv3(x)), dim=2)[0]
        return self.fc(torch.cat((x1, x2, x3), dim=1))


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

model     = TextCNN(vocab_size=tokenizer.vocab_size, embed_dim=128, num_classes=len(label_list)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

f1_scores = []

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["label"].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input_ids), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate after each epoch
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["label"].to(device)
            preds.extend(torch.argmax(model(input_ids), dim=1).cpu().numpy())
            true.extend(labels.cpu().numpy())

    acc = accuracy_score(true, preds)
    _, _, f1, _ = precision_recall_fscore_support(true, preds, average="weighted")
    f1_scores.append(f1)

    print(f"\nEpoch {epoch + 1}, Loss: {total_loss:.4f}")
    print(f"\nText-CNN Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}\n")

# ── Save F1 history ────────────────────────────────────────────────────────────
os.makedirs("experiments", exist_ok=True)
with open("experiments/cnn_f1_history.json", "w") as f:
    json.dump(f1_scores, f)

# ── Learning curve ─────────────────────────────────────────────────────────────
margin   = 0.01
all_vals = f1_scores

plt.figure()
plt.plot(range(1, len(f1_scores) + 1), f1_scores, marker='s')
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Text-CNN Learning Curve (F1 Score)")
plt.ylim(min(all_vals) - margin, min(max(all_vals) + margin, 1.0))
plt.xticks(range(1, len(f1_scores) + 1))
plt.savefig("experiments/f1_learning_curve_cnn.png")
plt.show()

# ── Confusion matrix ───────────────────────────────────────────────────────────
print("Generating confusion matrix...")

cm   = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)

fig, ax = plt.subplots(figsize=(9, 7))
disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
ax.set_title("Text-CNN Confusion Matrix")
plt.tight_layout()
plt.savefig("experiments/confusion_matrix_cnn.png")
plt.show()

print("\nTraining complete.")