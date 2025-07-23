import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
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
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(clf, "../app/ml/human_bot_classifier.joblib")
joblib.dump(vectorizer, "../app/ml/vectorizer.joblib")
print("Модель и векторизатор сохранены!")