import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pickle
from preprocess import preprocess_text

# Charger le dataset
df = pd.read_csv("../data/spam.csv")

# Prétraitement
df['message'] = df['message'].apply(preprocess_text)

# Train/Test split
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Modèles
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

# Entraînement et sauvegarde
best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train_vect, y_train)
    acc = model.score(X_test_vect, y_test)
    print(f"{name} Accuracy : {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = model

# Sauvegarder le meilleur modèle et le vectorizer
with open("../models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
