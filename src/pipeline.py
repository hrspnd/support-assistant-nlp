"""
pipeline.py  (lightweight — no external API, fully free)
Intent from BERT → MultiArmedBandit response selection → response.
"""

import re
import random
from dataclasses import dataclass, field
from typing import Optional

# ── RL agent ───────────────────────────────────────────────────────────────────
try:
    from rl_agent import MultiArmedBandit
except Exception:
    # Minimal epsilon-greedy fallback if rl_agent.py is unavailable
    class MultiArmedBandit:
        def __init__(self, n):
            self.n = n
            self.counts  = [0] * n
            self.values  = [0.0] * n
        def select_action(self):
            if random.random() < 0.1 or all(c == 0 for c in self.counts):
                return random.randrange(self.n)
            return max(range(self.n), key=lambda i: self.values[i])
        def update(self, action, reward):
            self.counts[action] += 1
            n = self.counts[action]
            self.values[action] += (reward - self.values[action]) / n

# ── BERT classifier ────────────────────────────────────────────────────────────
try:
    from predict import predict_intent
except Exception:
    _KW = {
        "track_order":       ["order", "where is my", "order status"],
        "track_delivery":    ["track", "tracking", "package", "shipment"],
        "delivery_time":     ["how long", "eta", "arrive", "expected", "days"],
        "missing_item":      ["missing", "not received", "wasn't there", "incomplete"],
        "damaged_delivery":  ["damaged", "broken", "crushed", "defective"],
        "delivery_issue":    ["problem", "issue", "wrong address", "failed"],
        "shipping_costs":    ["cost", "fee", "how much", "free shipping"],
    }
    def predict_intent(text: str) -> str:
        t = text.lower()
        for intent, kws in _KW.items():
            if any(k in t for k in kws):
                return intent
        return "delivery_issue"

# ── Pre-classifier checks ──────────────────────────────────────────────────────
_ESCALATION = ["speak to a human", "real person", "agent", "supervisor",
               "escalate", "refund", "compensation"]
_FAREWELL   = ["bye", "goodbye", "thanks bye", "thank you", "that's all",
               "no thanks", "nothing else"]
_GREETINGS  = ["hello", "hi", "hey", "good morning", "good afternoon",
               "good evening", "howdy", "what's up", "sup"]

def _is_escalation(text: str) -> bool:
    return any(p in text.lower() for p in _ESCALATION)

def _is_farewell(text: str) -> bool:
    t = text.lower().strip()
    return any(t == w or t.startswith(w) for w in _FAREWELL)

def _is_greeting(text: str) -> bool:
    t = text.lower().strip()
    return any(t == w or t.startswith(w) for w in _GREETINGS)

# ── Response variants (3 per intent for the bandit to choose between) ──────────
_RESPONSES = {
    "track_order": [
        "To track your order, visit our website and go to 'My Orders', "
        "or use the tracking link sent to your email when your order was dispatched. "
        "Orders are typically dispatched within 1–2 business days. "
        "Is there anything else I can help with?",

        "You can check your order status anytime under 'My Orders' on our website. "
        "A tracking link was also sent to your email once the order shipped. "
        "Is there anything else I can help with?",

        "Your order status is available in the 'My Orders' section of your account, "
        "or via the tracking link in your dispatch confirmation email. "
        "Dispatch usually happens within 1–2 business days. "
        "Is there anything else I can help with?",
    ],
    "track_delivery": [
        "You can track your delivery using the tracking number sent to your email. "
        "Enter it on our tracking page or the courier's website for real-time updates. "
        "Deliveries are updated every few hours. Is there anything else I can help with?",

        "Check your delivery status by entering your tracking number on our website or the courier's tracking page. "
        "You should have received the tracking number in your shipping confirmation email. "
        "Is there anything else I can help with?",

        "Your tracking number was included in the dispatch email — use it on our tracking page "
        "or the courier's site to see the latest delivery updates. "
        "Is there anything else I can help with?",
    ],
    "delivery_time": [
        "Standard delivery takes 3–5 business days. "
        "Express options (1–2 days) are available at checkout depending on your area. "
        "International orders may take 7–14 business days. "
        "Is there anything else I can help with?",

        "Most orders arrive within 3–5 business days with standard shipping. "
        "Express delivery (1–2 days) is available for eligible areas at checkout. "
        "Is there anything else I can help with?",

        "Delivery typically takes 3–5 business days for standard shipping, "
        "or 1–2 days if you selected express. International shipments take 7–14 business days. "
        "Is there anything else I can help with?",
    ],
    "missing_item": [
        "I'm sorry an item is missing from your order. "
        "Please report it through our website under 'My Orders' > 'Report an Issue', "
        "and our team will send the missing item or issue a refund within 24–48 hours. "
        "Is there anything else I can help with?",

        "Sorry to hear that — please go to 'My Orders' > 'Report an Issue' on our website "
        "to flag the missing item. We'll verify your order and resolve it within 24–48 hours. "
        "Is there anything else I can help with?",

        "Missing items can be reported under 'My Orders' > 'Report an Issue' on our website. "
        "Our team will review your order and either ship the missing item or process a refund promptly. "
        "Is there anything else I can help with?",
    ],
    "damaged_delivery": [
        "I'm sorry your item arrived damaged. "
        "Please take photos of the damage and packaging, then submit a report "
        "under 'My Orders' > 'Report an Issue'. Keep the packaging if possible — it helps with the claim. "
        "Our team will follow up within 24 hours. Is there anything else I can help with?",

        "I apologise for the damaged delivery. Go to 'My Orders' > 'Report an Issue' on our website, "
        "attach photos of the damage, and we'll get back to you within 24 hours. "
        "Please hold onto the packaging as it may be needed for the claim. "
        "Is there anything else I can help with?",

        "Sorry about that. Please document the damage with photos and report it via "
        "'My Orders' > 'Report an Issue'. Keeping the original packaging helps speed up the claim. "
        "We'll follow up within 24 hours. Is there anything else I can help with?",
    ],
    "delivery_issue": [
        "I'm sorry you're experiencing a delivery issue. "
        "Please check your tracking link for the latest status. "
        "If the problem persists, contact us through 'My Orders' > 'Report an Issue' and we'll investigate. "
        "Is there anything else I can help with?",

        "Sorry to hear that. First, check your tracking link for any status updates. "
        "If your package is marked delivered but hasn't arrived, please report it under "
        "'My Orders' > 'Report an Issue' and we'll look into it right away. "
        "Is there anything else I can help with?",

        "Please start by checking your tracking link for updates. "
        "If the issue isn't resolved there, use 'My Orders' > 'Report an Issue' on our website "
        "and our team will investigate promptly. Is there anything else I can help with?",
    ],
    "shipping_costs": [
        "Shipping costs depend on your location and order weight. "
        "Standard shipping is free on orders over $50. "
        "Express and international rates are calculated at checkout. "
        "Is there anything else I can help with?",

        "Standard shipping is free for orders over $50. "
        "For smaller orders or express/international delivery, rates are shown at checkout. "
        "Is there anything else I can help with?",

        "You can see exact shipping costs at checkout based on your location and order size. "
        "Standard shipping is free on orders over $50. "
        "Is there anything else I can help with?",
    ],
}

_ESCALATION_REPLY = (
    "I completely understand. I'm flagging your case for a human agent — "
    "reference: ESC-{ref}. Someone will be in touch within 1 business hour. "
    "Is there anything else I can note for them?"
)
_FAREWELL_REPLY = (
    "Thanks for reaching out! I hope everything gets sorted quickly. "
    "Don't hesitate to come back if you need anything else. 👋"
)

# ── State ──────────────────────────────────────────────────────────────────────
@dataclass
class ConversationState:
    escalation_ref: int  = 1000
    last_intent:    str  = None
    last_action:    int  = None

# ── Pipeline ───────────────────────────────────────────────────────────────────
class ConversationalPipeline:
    def __init__(self):
        self.state   = ConversationState()
        self.bandits = {}   # one bandit per intent, initialised on first use

    def _get_bandit(self, intent: str) -> MultiArmedBandit:
        if intent not in self.bandits:
            self.bandits[intent] = MultiArmedBandit(len(_RESPONSES[intent]))
        return self.bandits[intent]

    def record_reward(self, reward: float):
        """
        Call this externally to send a reward signal to the bandit for the
        last response that was shown. reward=1 for positive, 0 for negative.
        """
        if self.state.last_intent and self.state.last_action is not None:
            bandit = self._get_bandit(self.state.last_intent)
            bandit.update(self.state.last_action, reward)

    def respond(self, user_text: str) -> tuple[str, Optional[str]]:
        text = user_text.strip()

        if _is_farewell(text):
            return _FAREWELL_REPLY, None

        if _is_greeting(text):
            return "Hello! How can I help you with your delivery today?", None

        if _is_escalation(text):
            ref = self.state.escalation_ref
            self.state.escalation_ref += 1
            return _ESCALATION_REPLY.format(ref=ref), None

        intent  = predict_intent(text)
        bandit  = self._get_bandit(intent)
        action  = bandit.select_action()
        reply   = _RESPONSES[intent][action]

        # Simulate reward (replace with real feedback signal if available)
        reward  = 1 if random.random() < 0.7 else 0
        bandit.update(action, reward)

        # Remember for optional external reward call
        self.state.last_intent = intent
        self.state.last_action = action

        return reply, intent

    def reset(self):
        self.state = ConversationState()