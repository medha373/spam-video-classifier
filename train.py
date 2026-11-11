"""Spam Email Classifier: TF-IDF + Multinomial Naive Bayes (Enron email dataset)."""
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def clean_text(text) -> str:
    """Lowercase, remove URLs/HTML/non-letters, collapse spaces."""
    text = str(text) if text is not None else ""
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()


def main():
    print("Loading Enron spam email dataset...")
    data = load_dataset("SetFit/enron_spam")
    X_train = [clean_text(t) for t in data["train"]["text"]]
    y_train = data["train"]["label"]
    X_test = [clean_text(t) for t in data["test"]["text"]]
    y_test = data["test"]["label"]
    print(f"Train: {len(X_train)} emails | Test: {len(X_test)} emails")
    print(f"Spam share in train: {sum(y_train) / len(y_train):.1%}")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english", ngram_range=(1, 2), min_df=2,
                    max_features=100_000, sublinear_tf=True,
                ),
            ),
            ("clf", MultinomialNB(alpha=0.1)),
        ]
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    for k, v in metrics.items():
        print(f"{k.capitalize():<10}: {v:.4f}")
    print("\n", classification_report(y_test, preds, target_names=["ham", "spam"]))

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Ready-to-paste Results table for the README
    table = "| Metric    | Score |\n|-----------|-------|\n" + "\n".join(
        f"| {k.capitalize():<9} | {v * 100:.2f}% |" for k, v in metrics.items()
    )
    (OUT / "results_table.md").write_text(table + "\n")
    print("\nPaste this into README (also saved in outputs/results_table.md):\n")
    print(table)

    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=["ham", "spam"], cmap="Blues"
    )
    plt.title("Confusion Matrix - Enron Test Set")
    plt.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Words most associated with spam
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    words = vec.get_feature_names_out()
    diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    print("\nTop spam words:", list(words[diff.argsort()[-10:][::-1]]))

    joblib.dump(model, OUT / "spam_model.joblib")
    print("Saved model and results to outputs/")


if __name__ == "__main__":
    main()
