📩 Spam Detection using Machine Learning (English Only)
📌  Présentation du projet

Ce projet consiste à développer une application de détection automatique de messages spam en utilisant des techniques de Machine Learning et de traitement du langage naturel (NLP).
Le système est capable d’analyser des messages textuels en anglais uniquement et de prédire s’ils sont Spam ou Ham (Non-Spam).

Une application web interactive a été développée avec Streamlit et déployée sur Streamlit Cloud.

🎯 Objectifs du projet

Comprendre les étapes complètes d’un projet Machine Learning

Appliquer des techniques de NLP sur des données textuelles

Comparer et entraîner des modèles de classification

Déployer un modèle ML sous forme d’application web

Utiliser Git et GitHub pour la gestion de versions

🗂️ Structure du projet
spam-detection-ml/
│
├── data/
│   ├── spam.csv            # Dataset original
│   └── spam_clean.csv      # Dataset nettoyé
│
├── models/
│   ├── spam_model.pkl      # Modèle entraîné
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── exploration.ipynb   # Analyse exploratoire
│   ├── preprocessing.ipynb
│   └── modeling.ipynb
│
├── src/
│   ├── preprocess.py      # Fonctions de nettoyage du texte
│   ├── train.py           # Entraînement du modèle
│   ├── evaluate.py        # Évaluation du modèle
│   └── app.py             # Application Streamlit
│
├── requirements.txt
├── README.md
└── .gitignore

📊 Dataset

Nom : spam.csv

Type : messages SMS

Langue : Anglais

Classes :

spam

ham

Le dataset est nettoyé et sauvegardé sous spam_clean.csv.

🧹 Prétraitement des données

Les étapes de prétraitement incluent :

Conversion en minuscules

Suppression de la ponctuation et des caractères spéciaux

Suppression des stopwords (anglais)

Stemming (Porter Stemmer)

Suppression des valeurs manquantes (NaN)

🔠 Vectorisation

La transformation du texte est réalisée à l’aide de TF-IDF Vectorizer avec :

Unigrammes et bigrammes

Limitation du vocabulaire

Filtrage des mots trop rares ou trop fréquents

🤖 Modèle de Machine Learning

Algorithme choisi : Logistic Regression

Raisons :

Bonne performance sur les données textuelles

Rapide et interprétable

Adapté aux problèmes de classification binaire

Le modèle est entraîné sur les données prétraitées et sauvegardé pour une réutilisation ultérieure.

📈 Évaluation

Les métriques utilisées :

Accuracy

Precision

Recall

F1-score

Le modèle donne de bons résultats sur les données de test et généralise correctement.

🌐 Application Web (Streamlit)

L’application permet :

D’analyser un ou plusieurs messages (un par ligne)

D’afficher :

La prédiction (Spam / Ham)

La probabilité associée

De conserver un historique des messages analysés

🚀 Déploiement

L’application est déployée sur Streamlit Cloud et accessible via une URL publique.

Lancement en local :
streamlit run src/app.py

🧰 Technologies utilisées

Python

Pandas, NumPy

Scikit-learn

NLTK

Streamlit

Git & GitHub

⚠️ Limites du projet

Le modèle fonctionne uniquement pour des messages en anglais

Les messages en français ou autres langues peuvent être mal classés

Le dataset est relativement limité

🔮 Améliorations possibles

Support multilingue

Ajout d’un seuil personnalisable

Comparaison avec d’autres modèles (SVM, Naive Bayes)

Analyse des faux positifs / faux négatifs

Déploiement via API (FastAPI)

👨‍🎓 Auteur

Nom : Khalid

Projet académique en Machine Learning & NLP