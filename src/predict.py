from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = "models/bert_intent_classifier"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

labels = [
    "damaged_delivery",
    "delivery_issue",
    "delivery_time",
    "missing_item",
    "shipping_costs",
    "track_delivery",
    "track_order"
]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return labels[predicted_class]


while True:
    user_input = input("Customer message: ")
    print("Predicted intent:", predict(user_input))
