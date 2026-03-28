# plot_learning_curves.py
# Generates a combined BERT vs Text-CNN learning curve.
# Run this after both train_bert.py and train_cnn.py have completed.

import json
import os
import matplotlib.pyplot as plt

BERT_PATH = "experiments/bert_f1_history.json"
CNN_PATH  = "experiments/cnn_f1_history.json"
OUT_PATH  = "experiments/f1_learning_curve_combined.png"


def load(path):
    if not os.path.exists(path):
        print(f"Missing: {path} — run the corresponding training script first.")
        return []
    with open(path) as f:
        return json.load(f)


bert_f1 = load(BERT_PATH)
cnn_f1  = load(CNN_PATH)

if not bert_f1 and not cnn_f1:
    print("No data to plot.")
    exit()

plt.figure(figsize=(8, 5))

if bert_f1:
    plt.plot(range(1, len(bert_f1) + 1), bert_f1, marker='o', label="BERT")
if cnn_f1:
    plt.plot(range(1, len(cnn_f1) + 1), cnn_f1, marker='s', label="Text-CNN")

plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Learning Curve — BERT vs Text-CNN (F1 Score)")
all_vals = bert_f1 + cnn_f1
margin = 0.01

plt.ylim(min(all_vals) - margin, min(max(all_vals) + margin, 1.0))
plt.xticks(range(1, max(len(bert_f1), len(cnn_f1)) + 1))
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PATH)
plt.show()

print(f"Saved to {OUT_PATH}")