# Delivery Support Assistant (NLP + RL)

A hybrid customer support assistant that uses BERT and Text-CNN for intent classification, combined with a Multi-Armed Bandit (epsilon-greedy) for response selection.

---

## Quick Start

```bash
# 1. Install PyTorch (GPU)
python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Generate dataset (if missing)
python src/data_pipeline.py

# 4. Run the full pipeline
run_all.bat
```

---

## Prerequisites

Make sure you have **Python 3.10+** installed.

- [Download Python](https://www.python.org/downloads/) — pip is included by default
- To verify:
  ```bash
  python --version
  python -m pip --version
  ```

---

## Installation

### 1. Install PyTorch

**If you have an NVIDIA GPU (recommended):**
```bash
python -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bash
python -m pip install torch
```

> To check if your machine has a CUDA-compatible GPU, run `nvidia-smi` in your terminal.
> If the command is not found, use the CPU version.

### 2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

---

## Project Structure

Make sure your project follows this structure:

```
support-assistant-nlp/
├── data/
│   └── processed/
│       └── intent_dataset.csv
├── src/
│   ├── chatbot_ui.py
│   ├── eval.py
│   ├── pipeline.py
│   ├── plot_learning_curves.py
│   ├── predict.py
│   ├── train_bert.py
│   ├── train_cnn.py
│   ├── rl_agent.py
│   └── data_pipeline.py
├── experiments/
│   └── rl_evaluation.py
├── run_all.bat
├── requirements.txt
```

---

## Dataset

Ensure the dataset exists at:

```
data/processed/intent_dataset.csv
```

If the dataset is missing, generate it using:

```bash
python src/data_pipeline.py
```

---

## Run the Project

To execute the full pipeline (training + evaluation), run this command from the project root directory:
```bash
run_all.bat
```

---

## Run the Chatbot

After training is complete, launch the chatbot UI with:
```bash
python src/chatbot_ui.py
```

This will open the DeliverySupport AI desktop interface where you can interact with the trained assistant.

---

## Reproducibility

The entire system can be reproduced using a single command:

```bash
run_all.bat
```

This will:
- Train the BERT model
- Train the Text-CNN model
- Run reinforcement learning evaluation
- Generate performance graphs

- Launch the chatbot UI (`src/chatbot_ui.py`)

---

## Evaluation

### Model Comparison

Both models were evaluated under the same training conditions at 1 and 3 epochs.

| Model    | Epochs | Accuracy | F1 Score |
|----------|--------|----------|----------|
| Text-CNN | 1      | 99.08%   | 99.08%   |
| BERT     | 1      | 99.13%   | 99.08%   |
| Text-CNN | 3      | 99.16%   | 99.16%   |
| BERT     | 3      | 99.18%   | 99.16%   |

Both models exceed 99% accuracy in all configurations. BERT slightly outperforms Text-CNN, but the gap is minimal. For short, structured customer queries, Text-CNN offers a more computationally efficient alternative with comparable performance.

### Reinforcement Learning Results

The epsilon-greedy agent (ε = 0.5) initially explores broadly, causing reward fluctuation. Over time it converges toward the most rewarding action, demonstrating effective adaptive response selection. The high exploration rate slows convergence but ensures broader coverage of possible actions.

> Note: RL evaluation is based on simulated rewards and does not fully reflect real user interactions. The multi-armed bandit approach does not consider sequential context, which limits its ability to model multi-turn conversations.

### Ablation Study

**Model Architecture** — BERT and Text-CNN were trained under identical conditions. Both achieved high performance with a minimal accuracy gap, confirming that convolutional models are competitive with transformer-based models for this task.

**Training Epochs** — Increasing from 1 to 3 epochs produced slight improvements, but both models already achieved high accuracy after a single epoch, suggesting diminishing returns with additional training.

### Error Analysis

A small number of misclassifications were observed, mainly in two cases:

- **Ambiguous queries** — Short or vague inputs like "how much" were misclassified (predicted `track_delivery` instead of `shipping_costs`), while more specific queries like "how much is shipping" were correctly identified. The model relies heavily on explicit keywords.
- **Overlapping intents** — General statements like "there is a problem with my order" can correspond to multiple intent categories, making classification more difficult.
- **Vocabulary gaps** — Synonym variations not seen during training (e.g., "damaged" vs. "broken") can cause misclassification, highlighting the importance of dataset diversity.

---

## Notes

- The system uses a BERT-based model as the primary classifier and Text-CNN for comparison.
- A Multi-Armed Bandit (epsilon-greedy) is used for adaptive response selection.
- Reinforcement learning evaluation is based on simulated rewards, not real user interaction.

---

## Disclaimer

This project is developed for academic purposes only. The system may produce incorrect or incomplete responses and should not be used as a replacement for professional customer support.
