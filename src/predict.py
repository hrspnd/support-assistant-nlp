# predict.py
# Chatbot with BERT intent classification + RL response selection

import torch
import random
from transformers import BertTokenizer, BertForSequenceClassification
from rl_agent import MultiArmedBandit

# Load model and tokenizer
model_path = "models/bert_intent_classifier"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Label mapping (MUST match training)
label_list = [
    "damaged_delivery",
    "delivery_issue",
    "delivery_time",
    "missing_item",
    "shipping_costs",
    "track_delivery",
    "track_order"
]

# Responses per intent
intent_responses = {
    "damaged_delivery": [
        "I'm sorry your delivery arrived damaged. Could you describe the issue?",
        "I apologize for the damage. Let me help you report this.",
        "Please provide details about the damage so we can assist you."
    ],
    "delivery_issue": [
        "I'm sorry you're experiencing a delivery issue. Could you explain?",
        "Let me help resolve this delivery problem.",
        "Please provide more details about the issue."
    ],
    "delivery_time": [
        "Delivery time depends on location. Let me check for you.",
        "I can help estimate your delivery time.",
        "Please provide your order details for accurate timing."
    ],
    "missing_item": [
        "I'm sorry items are missing. Which ones?",
        "Let me help you report missing items.",
        "Please list the missing items so we can fix this."
    ],
    "shipping_costs": [
        "Shipping costs depend on location and order size.",
        "I can help you check shipping fees.",
        "Let me assist you in finding the exact shipping cost."
    ],
    "track_delivery": [
        "Please provide your tracking number.",
        "Let me check your delivery status.",
        "I can help track your package."
    ],
    "track_order": [
        "Please provide your order number.",
        "Let me check your order status.",
        "I can help track your order."
    ]
}

# RL agents per intent
bandits = {}


def predict_intent(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return label_list[predicted_class]


def get_response(intent):
    responses = intent_responses[intent]

    # initialize bandit if not exists
    if intent not in bandits:
        bandits[intent] = MultiArmedBandit(len(responses))

    agent = bandits[intent]

    action = agent.select_action()
    response = responses[action]

    # simulate reward (you can tweak this)
    reward = 1 if random.random() < 0.7 else 0

    agent.update(action, reward)

    return response


def chatbot():
    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        intent = predict_intent(user_input)
        response = get_response(intent)

        print(f"Intent: {intent}")
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    chatbot()