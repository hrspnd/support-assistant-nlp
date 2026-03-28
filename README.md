# Delivery Customer Support Assistant (NLP + RL)

A hybrid customer support assistant that uses BERT and Text-CNN for intent classification, combined with a Multi-Armed Bandit (epsilon-greedy) for response selection.

---

## Development Team
Bermudo, Jeanne Clarisse T.

Magat, Maria Josephine M.

Pineda, Mary Alexa Ysabelle V.

Rebusa, Amber Kaia J.

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

---

## Notes

- The system uses a BERT-based model as the primary classifier and Text-CNN for comparison.
- A Multi-Armed Bandit (epsilon-greedy) is used for adaptive response selection.
- Reinforcement learning evaluation is based on simulated rewards, not real user interaction.

---

## Disclaimer

This project is developed for academic purposes only. The system may produce incorrect or incomplete responses and should not be used as a replacement for professional customer support.

