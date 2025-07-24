import re
import os 
import joblib
import numpy as np

def regex_rule_model(text: str) -> int:
    patterns = [
        r"\bя не бот\b",
        r"\bваш запрос принят\b",
        r"\bя искусственный интеллект\b",
        r"\bобработка запроса\b",
        r"\bуточните ваш вопрос\b"
    ]
    for pat in patterns:
        if re.search(pat, text.lower()):
            return 1 
    return 0  

def keyword_rule_model(text: str) -> int:
    keywords = [
        "запрос", "обработка", "бот", "искусственный интеллект", "пожалуйста, уточните", "ваш вопрос"
    ]
    for kw in keywords:
        if kw in text.lower():
            return 1
    return 0

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "app", "ml", "vectorizer.joblib")
MODEL_LR_PATH = os.path.join(PROJECT_ROOT, "app", "ml", "human_bot_classifier.joblib")
MODEL_RF_PATH = os.path.join(PROJECT_ROOT, "app", "ml", "human_bot_classifier_rf.joblib")

vectorizer = joblib.load(VECTORIZER_PATH)
clf_lr = joblib.load(MODEL_LR_PATH)
clf_rf = joblib.load(MODEL_RF_PATH)

def ml_model_lr(text: str) -> int:
    vec = vectorizer.transform([text])
    return int(clf_lr.predict(vec)[0])

def ml_model_rf(text: str) -> int:
    vec = vectorizer.transform([text])
    return int(clf_rf.predict(vec)[0])

"""Пишу ансамбль"""
def ensemble_predict(text: str, threshold=0.3, weights=None) -> int:
    preds = [
        regex_rule_model(text),
        keyword_rule_model(text),
        ml_model_lr(text),
        ml_model_rf(text)
    ]
    if weights is None:
        prob = np.mean(preds)
    else:
        prob = np.average(preds, weights=weights)
    return int(prob > threshold)

def get_bot_probability_ensemble(text: str) -> int:
    return ensemble_predict(text)