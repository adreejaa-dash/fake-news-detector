# Fake News Detection System

An NLP-powered pipeline that classifies news article text as **Fake** or **Real**, with a confidence score, using TF-IDF features and three trained classifiers.

## Problem Statement

Misinformation spreads rapidly online. This project builds a machine-learning pipeline that ingests raw article text, applies a reproducible NLP preprocessing pipeline (identical at training and inference time), and outputs a binary label with confidence. The goal is a well-evaluated ML pipeline, not a polished UI.

## Dataset

- **Source**: Kaggle — combined LIAR + GossipCop fake news dataset (preprocessed)
- **File**: `data/news_full.csv`
- **Size**: ~44,000 articles
- **Features used**: `text` (raw article body) → cleaned and TF-IDF vectorized
- **Target**: `label_number` (0 = Fake, 1 = Real)

## NLP Pipeline

1. Lowercase, strip URLs, punctuation, special characters
2. NLTK tokenization + stopword removal
3. TF-IDF vectorization — unigrams + bigrams, `max_features=8000`, `sublinear_tf=True`

## Model Comparison

| Model                   | Accuracy | Precision | Recall | F1     |
|-------------------------|----------|-----------|--------|--------|
| Logistic Regression     | 0.9901   | 0.9867    | 0.9955 | 0.9911 |
| Multinomial Naive Bayes | 0.9577   | 0.9534    | 0.9705 | 0.9618 |
| **XGBoost** ✅ (saved)  | **0.9955** | **0.9937** | **0.9981** | **0.9959** |

> Dataset: 38,516 articles | 80/20 train-test split | TF-IDF unigrams+bigrams (max 8,000 features)

## Setup & Run

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Train models (saves model.pkl, vectorizer.pkl)
python train.py

# 3. Start the Flask inference app
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) and paste any news article text to get a prediction.
