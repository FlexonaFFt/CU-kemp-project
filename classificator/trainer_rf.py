import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

engine = create_engine('sqlite:///../chat_history.db')
query = "SELECT role, content FROM messages"
df = pd.read_sql(query, engine)
df = df[df['role'].isin(['user', 'assistant'])]
df['label'] = df['role'].apply(lambda x: 1 if x == 'user' else 0)

X = df['content'].astype(str)
y = df['label']
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)

joblib.dump(clf_rf, "../app/ml/human_bot_classifier_rf.joblib")
joblib.dump(vectorizer, "../app/ml/vectorizer.joblib") 
print("RandomForest модель и векторизатор сохранены!")
