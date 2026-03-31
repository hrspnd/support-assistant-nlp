## Ethics & Policy Statement

This system is developed to assist with delivery-related customer support queries using machine learning models. While it aims to improve efficiency and response quality, several ethical considerations are addressed.

### Risks and Mitigation

One primary risk is incorrect intent classification, which may result in irrelevant or misleading responses. To mitigate this, the models are evaluated using accuracy and F1-score, and responses are constrained to predefined templates. Additionally, a reinforcement learning component improves response selection over time based on feedback signals.

### Privacy

The system processes user-generated text, which may contain sensitive information. Only publicly available and properly licensed datasets are used for training. No personally identifiable information (PII) is stored. In deployment, input filtering and anonymization mechanisms are recommended to further protect user privacy.

### Fairness

Bias may exist in the dataset, especially if certain intents or language styles are overrepresented. To address this, the dataset is reviewed for class balance, and performance is monitored across all intent categories. However, the system may still perform less effectively on rare or ambiguous queries.

### Intended Use

The system is intended for handling common delivery-related inquiries such as order tracking, delivery issues, and missing items. It is designed as a support tool and not as a replacement for human customer service.

### Limitations

The system is limited by the quality and scope of its training data. It may not generalize well beyond predefined intents and cannot fully understand complex or nuanced queries. The reinforcement learning component is also limited to simple decision-making and does not model long-term conversation context.

Users should be informed that responses are AI-generated and may not always be accurate. A fallback to human support is recommended for critical or complex concerns.

