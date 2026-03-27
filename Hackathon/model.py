# -----------------------------
# Import Libraries
# -----------------------------
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("complete_dataset.csv")  # columns: text, label
# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess(text):
    text = str(text).lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return text
df["text"] = df["text"].apply(preprocess)
# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["text"])

y = df["label"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model 1: Logistic Regression
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)


from xgboost import XGBClassifier
# -----------------------------
# Model 2: XGBoost
# -----------------------------
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
# -----------------------------
# Evaluation
# -----------------------------
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))


print("\n--- Logistic Regression Report ---")
print(classification_report(y_test, lr_pred))

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

print("\n--- XGBoost Report ---")
print(classification_report(y_test, xgb_pred))

# -----------------------------
# User Input Prediction
# -----------------------------
while True:
    user_text = input("\nEnter text (or type 'exit' to quit): ")

    if user_text.lower() == "exit":
        break

    # Preprocess
    processed_text = preprocess(user_text)

    # Transform using SAME vectorizer
    vectorized_text = vectorizer.transform([processed_text])

    # -----------------------------
    # Logistic Regression Prediction
    # -----------------------------
    lr_prediction = lr_model.predict(vectorized_text)[0]
    lr_probs = lr_model.predict_proba(vectorized_text)[0]

    # Get confidence (max probability)
    lr_confidence = max(lr_probs)

    print("\n[Logistic Regression]")
    print("Prediction:", lr_prediction)
    print("Confidence:", round(lr_confidence * 100, 2), "%")

    # -----------------------------
    # XGBoost Prediction
    # -----------------------------
    xgb_prediction = xgb_model.predict(vectorized_text)[0]
    xgb_probs = xgb_model.predict_proba(vectorized_text)[0]

    xgb_confidence = max(xgb_probs)

    print("\n[XGBoost]")
    print("Prediction:", xgb_prediction)
    print("Confidence:", round(xgb_confidence * 100, 2), "%")

# 0 - Human
# 1 - AI
