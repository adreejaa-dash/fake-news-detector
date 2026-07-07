import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"[WARN] XGBoost not available ({e}). Using RandomForest instead.")
    XGBOOST_AVAILABLE = False

from preprocess import clean_text

print("Loading data...")
df = pd.read_csv("data/news_full.csv", usecols=["text", "label_number"])
df = df.dropna(subset=["text", "label_number"])
df["label_number"] = df["label_number"].astype(int)

print(f"Dataset size: {len(df)} rows  |  Label distribution:\n{df['label_number'].value_counts().to_string()}")

print("Preprocessing text (this may take a while)...")
df["processed"] = df["text"].apply(clean_text)

X = df["processed"]
y = df["label_number"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

if XGBOOST_AVAILABLE:
    third_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 eval_metric="logloss",
                                 tree_method="hist", n_jobs=-1, random_state=42)
    third_name = "XGBoost"
else:
    third_model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    third_name = "Random Forest"

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    third_name: third_model,
}

results = {}

print("\n" + "="*70)
for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    cm   = confusion_matrix(y_test, preds)

    results[name] = {"model": model, "accuracy": acc, "precision": prec,
                     "recall": rec, "f1": f1}

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
print("="*70)

best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]
best_acc   = results[best_name]["accuracy"]
print(f"\nBest model: {best_name}  (F1={results[best_name]['f1']:.4f})")

print("Saving model and vectorizer...")
joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Always save LR separately so Flask can use its coef_ for top-word explanations
joblib.dump(results["Logistic Regression"]["model"], "lr_model.pkl")

model_info = {
    "name": best_name,
    "accuracy": best_acc,
    "all_results": {k: {m: v for m, v in r.items() if m != "model"} for k, r in results.items()}
}
joblib.dump(model_info, "model_info.pkl")

print("Done! model.pkl, vectorizer.pkl, model_info.pkl saved.")
print("\nFull comparison:")
print(f"{'Model':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-"*58)
for name, r in results.items():
    print(f"{name:<30} {r['accuracy']:>6.4f} {r['precision']:>6.4f} {r['recall']:>6.4f} {r['f1']:>6.4f}")
