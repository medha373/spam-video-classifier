"""Classify custom email text as spam or ham.

Usage:
    python predict.py "Congratulations! You won a free prize, click here now"
    python predict.py            # runs built-in examples
"""
import sys

import joblib

from train import OUT, clean_text

EXAMPLES = [
    "Congratulations! You have won a free prize. Click here to claim your reward now!!!",
    "Hi team, please find attached the agenda for tomorrow's project meeting.",
    "Cheap meds online, no prescription needed, limited time offer",
]


def predict(model, text: str):
    proba = model.predict_proba([clean_text(text)])[0]
    label = "SPAM" if proba[1] >= 0.5 else "NOT SPAM"
    return label, max(proba)


if __name__ == "__main__":
    model = joblib.load(OUT / "spam_model.joblib")
    texts = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else EXAMPLES
    for t in texts:
        label, conf = predict(model, t)
        print(f"{label} ({conf:.0%}) -> {t}")
