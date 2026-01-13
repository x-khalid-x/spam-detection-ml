import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Charger modèle et vectorizer
with open("../models/best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("../models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Charger test set (X_test et y_test)
import pandas as pd
df = pd.read_csv("../data/spam_clean.csv")
X_test = df['message'].fillna("")
y_test = df['label']

X_test_vect = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vect)

# Rapport
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['ham','spam'], yticklabels=['ham','spam'])
plt.show()
