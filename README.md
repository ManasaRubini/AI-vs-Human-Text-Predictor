# AI-vs-Human-Text-Predictor
AI vs Human Text Detector using a hybrid ML approach (TF-IDF + XGBoost) with linguistic and behavioral features like entropy, burstiness, and typing speed.

##  Features Used
- TF-IDF (Text representation)
- Logistic Regression (Base model)
- XGBoost (Meta model)
- Burstiness (sentence variation)
- Repetition Score
- Word Entropy
- Typing Behavior (speed-based detection)

##  Workflow
1. Text is converted into TF-IDF features
2. Logistic model predicts probability
3. Additional features are extracted:
   - Burstiness
   - Repetition
   - Entropy
4. XGBoost combines all features
5. Behavioral feature (typing speed) is added

##  Output
- Prediction: AI or Human
- Confidence Score
- Behavioral Insight

##  How to Run

```bash
pip install -r requirements.txt
python app.py
