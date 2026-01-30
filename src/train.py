import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_text
# 1. Charger les données
df = pd.read_csv("data/spam_clean.csv")
# Supprimer NaN et messages trop courts
df = df.dropna(subset=["Message", "Category"])
df["Message"] = df["Message"].astype(str)
df = df[df["Message"].str.strip().str.len() > 3]
# 2. Prétraitement
X = df["Message"].apply(clean_text)
y = df["Category"]
# 3. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 4. TF-IDF 
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
# 5. Modèle Logistic Regression
model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train_vect, y_train)
# 6. Évaluation rapide
y_pred = model.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# 7. Sauvegarde
with open("models/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("Modèle et vectorizer sauvegardés")
