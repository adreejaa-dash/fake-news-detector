import os
import joblib
from flask import Flask, render_template, request
from preprocess import clean_text

app = Flask(__name__)

print("Loading model and vectorizer...")
model      = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
model_info = joblib.load("model_info.pkl")
model_name = model_info["name"]
model_acc  = model_info["accuracy"]

# Load LR model for top-word explanations (has coef_), fallback to best model
lr_model = joblib.load("lr_model.pkl") if os.path.exists("lr_model.pkl") else model

TOP_WORDS = 10


def get_top_words(text_vec, predicted_class):
    if not hasattr(lr_model, "coef_"):
        return []
    feature_names = vectorizer.get_feature_names_out()
    coef = lr_model.coef_[0]
    scores = text_vec.toarray()[0] * coef
    if predicted_class == 0:
        scores = -scores
    top_indices = scores.argsort()[-TOP_WORDS:][::-1]
    return [(feature_names[i], round(float(scores[i]), 4)) for i in top_indices if scores[i] > 0]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        article_text = request.form.get("article_text", "").strip()
        if article_text:
            processed = clean_text(article_text)
            vec = vectorizer.transform([processed])
            prediction = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            confidence = round(float(proba[prediction]) * 100, 2)
            label = "Real" if prediction == 1 else "Fake"
            top_words = get_top_words(vec, prediction)
            result = {
                "label": label,
                "confidence": confidence,
                "model": model_name,
                "model_accuracy": round(model_acc * 100, 2),
                "top_words": top_words,
            }
    return render_template("index.html", result=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port)
