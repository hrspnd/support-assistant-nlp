import matplotlib.pyplot as plt

# Replace these with your actual results
models = ["BERT", "Text-CNN"]
f1_scores = [0.91, 0.87]

plt.figure()
plt.bar(models, f1_scores)

plt.xlabel("Models")
plt.ylabel("F1 Score")
plt.title("Model Comparison (F1 Score)")
plt.ylim(0, 1)

# Add value labels on top
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.savefig("experiments/f1_comparison.png")
plt.show()