# Spam Email Classifier

A machine learning model that classifies emails as **spam** or **not spam (ham)** using NLP.

## Overview
- **Dataset:** [Enron Spam Emails](https://huggingface.co/datasets/SetFit/enron_spam) (labelled spam / ham emails, with a train and test split)
- **Text preprocessing:** lowercasing, URL and HTML removal, punctuation/number removal, English stop-word removal
- **Features:** TF-IDF with unigrams and bigrams
- **Model:** Multinomial Naive Bayes
- **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Project Structure
```
spam-email-classifier/
├── train.py            # Load data, train, evaluate, save model
├── predict.py          # Classify custom email text
├── requirements.txt
├── outputs/
│   ├── metrics.json
│   ├── results_table.md
│   └── confusion_matrix.png
└── README.md
```

## How to Run
```bash
git clone https://github.com/medha373/spam-email-classifier.git
cd spam-email-classifier
pip install -r requirements.txt
python train.py
python predict.py "Congratulations! You won a free prize, click here"
```

## Results
(Paste the table from `outputs/results_table.md` here after running `train.py`.)

![Confusion Matrix](outputs/confusion_matrix.png)

## Why these metrics?
In spam detection, **precision** matters because wrongly flagging a real email as spam is costly, and **recall** matters because missed spam reaches the inbox. F1 balances both.

## Key Learnings
- TF-IDF + Naive Bayes is a fast, strong baseline for text classification.
- Bigrams capture phrases like "click here" and "free offer".
- Accuracy alone can mislead, so precision and recall are reported too.

## Future Improvements
- Compare with Logistic Regression and SVM
- Use email subject and body separately
- Deploy as a Streamlit web app

## Tech Stack
Python, scikit-learn, Hugging Face Datasets, Matplotlib, Joblib
