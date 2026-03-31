\# Model Card: Delivery Support Assistant (NLP \+ RL)

\#\# Model Overview

This project implements a hybrid customer support assistant that combines:

\- \*\*BERT (bert-base-uncased)\*\* for primary intent classification

\- \*\*Text-CNN\*\* as a comparative model

\- \*\*Multi-Armed Bandit (epsilon-greedy)\*\* for response selection

The system is designed to classify delivery-related customer queries and provide appropriate predefined responses.

\---

\#\# Intended Use

The model is intended for:

\- Automating delivery-related customer support

\- Handling queries such as:

  \- Order tracking

  \- Delivery issues

  \- Missing items

  \- Shipping concerns

It performs best on short, structured queries within predefined intent categories.

\---

\#\# Not Intended For

This system is not designed for:

\- Complex multi-turn conversations

\- Open-domain dialogue

\- Legal, medical, or financial advice

\- Processing sensitive personal data

\- Fully replacing human customer support

\---

\#\# Model Details

\#\#\# BERT Model

| Parameter | Value |

|---|---|

| Pretrained model | \`bert-base-uncased\` |

| Task | Intent classification |

| Max sequence length | 32 |

| Optimizer | AdamW |

| Epochs | 3 |

| Batch size | 32 |

\#\#\# Text-CNN Model

| Parameter | Value |

|---|---|

| Embedding dimension | 128 |

| Convolution filters | 3, 4, 5 (100 filters each) |

| Optimizer | Adam |

| Epochs | 3 |

| Batch size | 32 |

\#\#\# Reinforcement Learning Component

| Parameter | Value |

|---|---|

| Algorithm | Multi-Armed Bandit (epsilon-greedy) |

| Exploration rate (ε) | 0.5 |

| Update method | Incremental action-value estimation |

\---

\#\# Training Data

\- \*\*Source:\*\* Hugging Face — Bitext retail/e-commerce dataset

\- \*\*License:\*\* Community Data License Agreement – Sharing, Version 1.0

\- \*\*Data type:\*\* Labeled customer support queries

\- \*\*Preprocessing:\*\*

  \- Tokenization using BERT tokenizer

  \- Label encoding

  \- Padding and truncation to max length 32

\---

\#\# Evaluation

\#\#\# Metrics

Accuracy, Precision, Recall, and F1-score (weighted average).

\#\#\# Results

| Model    | Epochs | Accuracy | F1 Score |

|----------|--------|----------|----------|

| Text-CNN | 1      | 99.08%   | 99.08%   |

| BERT     | 1      | 99.13%   | 99.08%   |

| Text-CNN | 3      | 99.16%   | 99.16%   |

| BERT     | 3      | 99.18%   | 99.16%   |

\- BERT achieves marginally higher performance than Text-CNN across all configurations

\- Text-CNN provides a lightweight, computationally efficient baseline

\- The RL component improves response selection but does not affect classification metrics

\---

\#\# Limitations

\- Sensitive to vocabulary coverage (e.g., \`damaged\` vs. \`broken\`)

\- Limited to predefined intent classes

\- Performance depends on dataset quality and balance

\- RL component does not model multi-turn conversations

\- Uses simulated rewards instead of real user feedback

\---

\#\# Ethical Considerations

\- Potential bias from dataset imbalance

\- Risk of incorrect predictions on ambiguous or out-of-scope queries

\- Requires safeguards and human oversight for responsible deployment

\---

\#\# Deployment Guidance

\- Use human fallback for uncertain or escalated cases

\- Apply confidence thresholds to trigger escalation

\- Monitor system performance and logs regularly

\- Retrain periodically with updated and diverse data

\- Apply input filtering for safety

\---

\#\# Reproducibility

Run the full pipeline using:

\`\`\`bash

run\_all.bat

\`\`\`

Dataset must exist at:

\`\`\`

data/processed/intent\_dataset.csv

\`\`\`

If missing, generate it with:

\`\`\`bash

python src/data\_pipeline.py

\`\`\`

\---

\#\# Disclaimer

This system generates automated responses and may produce incorrect or incomplete outputs. It is intended for academic use only and should not replace human customer support.

