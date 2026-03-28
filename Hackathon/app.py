from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import pickle
import numpy as np
import math
from collections import Counter
import webbrowser

app = Flask(__name__)
CORS(app)

# =============================
# Load Models
# =============================
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
log_model = pickle.load(open("log_model.pkl", "rb"))
meta_model = pickle.load(open("meta_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =============================
# Feature Functions
# =============================
def burstiness(text):
    sentences = text.split('.')
    lengths = [len(s.split()) for s in sentences if len(s.strip()) > 0]
    return np.std(lengths) if lengths else 0

def repetition_score(text):
    words = text.lower().split()
    return len(words) / len(set(words)) if len(words) > 0 else 0

def word_entropy(text):
    words = text.lower().split()
    if len(words) == 0:
        return 0

    counts = Counter(words)
    total = len(words)

    entropy = 0
    for word in counts:
        p = counts[word] / total
        entropy -= p * math.log2(p)

    return entropy

# =============================
# Prediction
# =============================
def predict_text(text):
    tfidf_vec = vectorizer.transform([text])
    log_prob = log_model.predict_proba(tfidf_vec)[0][1]

    burst = burstiness(text)
    repeat = repetition_score(text)
    entropy = word_entropy(text)

    feats = [log_prob, burst, repeat, entropy]
    feats = scaler.transform([feats])

    pred = meta_model.predict(feats)[0]
    prob = meta_model.predict_proba(feats)[0][1]

    if pred == 1:
        return "AI Generated", prob
    else:
        return "Human Written", 1 - prob

# =============================
# Routes
# =============================

# Redirect root → HTML page
from flask import Flask, render_template, redirect, url_for
app = Flask(__name__)

# Root → redirect to landing page
@app.route("/")
def root():
    return redirect(url_for("index"))

# Landing page (index.html)
@app.route("/ui")
def index():
    return render_template("index.html")

# Home page (home.html)
@app.route("/home")
def home_page():
    return render_template("home.html")
# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    text = data.get("text", "")
    time_taken = data.get("time", 0)

    result, confidence = predict_text(text)

    # Optional behavior logic
    words = len(text.split())

    # avoid division by zero
    time_taken = max(time_taken, 0.5)

    wpm = (words / time_taken) * 60  # words per minute

    # smarter logic
    if time_taken < 3:
        behavior = "Possible Copy-Paste (AI-like)"
    elif wpm > 80:
        behavior = "Very Fast Typing (Suspicious)"
    elif wpm < 20:
        behavior = "Slow Typing (Human-like)"
    else:
        behavior = "Normal Typing (Human-like)"
    if confidence >= 0.8:
        decision = "Acceptable (High certainty)"
    elif 0.6 <= confidence < 0.8:
        decision = "Needs Review (Moderate certainty)"
    else:
        decision = "Likely AI-generated / Uncertain"

    return jsonify({
    "label": str(result),
    "confidence": float(round(float(confidence), 2)),
    "decision": decision, 
    "behavior": str(behavior)
})

# =============================
# Run + Auto Open Browser
# =============================
if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)
