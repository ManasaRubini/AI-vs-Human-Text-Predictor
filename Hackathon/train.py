# =============================
# AI vs Human Text Detector
# FINAL FIXED VERSION (Fast + Accurate)
# =============================

import numpy as np
import pandas as pd
import pickle
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

# =============================
# Load Dataset
# =============================
df = pd.read_csv("complete_dataset.csv")
df = df.dropna()

texts = df['text']
labels = df['label'].astype(int)

# =============================
# TF-IDF Features
# =============================
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(texts)

# =============================
# Base Model (Logistic Regression)
# =============================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_tfidf, labels)


# =============================
# Feature Functions
# =============================
import math
from collections import Counter

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
# Extract Features
# =============================

def extract_features(text):
    tfidf_vec = vectorizer.transform([text])

    log_prob = log_model.predict_proba(tfidf_vec)[0][1]

    burst = burstiness(text)
    repeat = repetition_score(text)
    entropy = word_entropy(text)

    return [log_prob, burst, repeat, entropy]

# =============================
# Prepare Meta Features
# =============================
X_features = []

for i, text in enumerate(texts):
    if i % 100 == 0:
        print(f"Processing: {i}/{len(texts)}")

    X_features.append(extract_features(text))

X_features = np.array(X_features)

# Normalize
scaler = MinMaxScaler()
X_features = scaler.fit_transform(X_features)

# =============================
# Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_features, labels, test_size=0.2, random_state=42
)

# =============================
# XGBoost Meta Model
# =============================
meta_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

meta_model.fit(X_train, y_train)

# =============================
# Evaluation
# =============================
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = meta_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Overall Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# =============================
# Save Models
# =============================
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(log_model, open("log_model.pkl", "wb"))
pickle.dump(meta_model, open("meta_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Models Saved Successfully")

# =============================
# Prediction Function
# =============================

def predict_text(text):
    feats = extract_features(text)
    feats = scaler.transform([feats])

    pred = meta_model.predict(feats)[0]
    prob = meta_model.predict_proba(feats)[0][1]

    if pred == 1:
        return f"AI Generated (Confidence: {prob:.2f})"
    else:
        return f"Human Written (Confidence: {1 - prob:.2f})"

# =============================
# Input
# =============================
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter text (or type 'exit' to quit):\n")

        if user_input.lower() == 'exit':
            break

        result = predict_text(user_input)
        print("\nResult:", result)