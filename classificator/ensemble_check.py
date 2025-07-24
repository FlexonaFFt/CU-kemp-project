import pandas as pd
from sqlalchemy import create_engine
from ensemble import get_bot_probability_ensemble, ensemble_predict, regex_rule_model, keyword_rule_model, ml_model_lr, ml_model_rf

engine = create_engine('sqlite:///../chat_history.db')
query = "SELECT id, role, content FROM messages"
df = pd.read_sql(query, engine)
df = df[df['role'].isin(['user', 'assistant'])]
df['label'] = df['role'].apply(lambda x: 1 if x == 'user' else 0)

results = []
for idx, row in df.iterrows():
    text = row['content']
    label = row['label']
    regex_pred = regex_rule_model(text)
    keyword_pred = keyword_rule_model(text)
    lr_pred = ml_model_lr(text)
    rf_pred = ml_model_rf(text)
    ensemble_prob = ensemble_predict(text)
    ensemble_pred = int(ensemble_prob > 0.5)
    results.append({
        "id": row['id'],
        "text": text,
        "label": label,
        "regex_pred": regex_pred,
        "keyword_pred": keyword_pred,
        "lr_pred": lr_pred,
        "rf_pred": rf_pred,
        "ensemble_prob": ensemble_prob,
        "ensemble_pred": ensemble_pred
    })

results_df = pd.DataFrame(results)

from sklearn.metrics import classification_report, accuracy_score

print("=== Ensemble Classification Report ===")
print(classification_report(results_df['label'], results_df['ensemble_pred']))
print("Accuracy:", accuracy_score(results_df['label'], results_df['ensemble_pred']))

results_df.to_csv("ensemble_predictions.csv", index=False)
print("Результаты сохранены в ensemble_predictions.csv")