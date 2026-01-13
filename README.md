# Projet Spam Detection ML

## Description
Détection de messages spam en utilisant du Machine Learning avec trois modèles :  
- Naive Bayes  
- Logistic Regression  
- Support Vector Machine (SVM)

Le projet inclut : exploration des données, prétraitement, entraînement des modèles, comparaison, tests avec de nouveaux messages et sauvegarde du meilleur modèle.

---

## Structure du projet
spam-detection-ml/
│
├── data/ # Dataset
├── notebooks/ # Notebooks : exploration, preprocessing, modeling
├── src/ # Scripts Python : prétraitement, entraînement, évaluation, prédiction
├── models/ # Modèles et vectorizer sauvegardés
├── README.md
├── requirements.txt
└── .gitignore

---

## Étapes du projet
1. Exploration des données  
2. Prétraitement du texte (cleaning, stopwords, stemming)  
3. Vectorisation TF-IDF  
4. Entraînement de 3 modèles ML  
5. Comparaison et choix du meilleur modèle  
6. Tests avec de nouveaux messages  
7. Sauvegarde du modèle et du vectorizer  

---

## Instructions pour lancer

### Installer les dépendances
```bash
pip install -r requirements.txt
# Exploration des données
jupyter notebook notebooks/exploration.ipynb
# Prétraitement du texte
jupyter notebook notebooks/preprocessing.ipynb
# Entraînement et modélisation
jupyter notebook notebooks/modeling.ipynb
python src/predict_new.py
## Auteur 
khalid

