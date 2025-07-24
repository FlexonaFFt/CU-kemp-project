import pandas as pd
from sqlalchemy import create_engine
from model import ml_model_lr, ml_model_rf, llm_predict_1, llm_predict_2, get_bot_probability_ensemble

engine = create_engine('sqlite:///../../chat_history.db')
query = "SELECT id, role, content FROM messages"
df = pd.read_sql(query, engine)
df = df[df['role'].isin(['user', 'assistant'])]
df['label'] = df['role'].apply(lambda x: 1 if x == 'assistant' else 0)
df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

results = []
for idx, row in df.iterrows():
    text = row['content']
    label = row['label']
    pred_lr = ml_model_lr(text)
    pred_rf = ml_model_rf(text)
    pred_llm1 = llm_predict_1(text)
    pred_llm2 = llm_predict_2(text)
    ensemble_prob = get_bot_probability_ensemble(text)
    ensemble_pred = int(ensemble_prob > 0.5)
    results.append({
        "id": row['id'],
        "text": text,
        "label": label,
        "ml_lr": pred_lr,
        "ml_rf": pred_rf,
        "llm1": pred_llm1,
        "llm2": pred_llm2,
        "ensemble_prob": ensemble_prob,
        "ensemble_pred": ensemble_pred
    })

results_df = pd.DataFrame(results)
from sklearn.metrics import classification_report, accuracy_score

print("=== Ensemble Classification Report ===")
print(classification_report(results_df['label'], results_df['ensemble_pred']))
print("Accuracy:", accuracy_score(results_df['label'], results_df['ensemble_pred']))
results_df.to_csv("ensemble_llm_predictions.csv", index=False)
print("Результаты сохранены в ensemble_llm_predictions.csv")
