import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import clean_text

# ===============================
# 1. Charger les données
# ===============================
df = pd.read_csv("data/spam_clean.csv")

# Sécurité ABSOLUE NLP
df = df.dropna(subset=["Message", "Category"])
df["Message"] = df["Message"].astype(str)
df = df[df["Message"].str.strip().str.len() > 3]

# ===============================
# 2. Nettoyage (OBLIGATOIRE)
# ===============================
X_clean = df["Message"].apply(clean_text)
y = df["Category"]

# ===============================
# 3. Charger modèle & vectorizer
# ===============================
with open("models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ===============================
# 4. Vectorisation + Prédiction
# ===============================
X_vect = vectorizer.transform(X_clean)
preds = model.predict(X_vect)

# ===============================
# 5. Résultats
# ===============================
print("Confusion Matrix:\n")
print(confusion_matrix(y, preds))

print("\nClassification Report:\n")
print(classification_report(y, preds))
