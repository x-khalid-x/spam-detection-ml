import pickle

# Charger modèle et vectorizer
with open("../models/best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("../models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def predict_spam(message):
    vect = vectorizer.transform([message])
    return model.predict(vect)[0]

# Exemple
if __name__ == "__main__":
    msg = input("Entrez un message : ")
    print("Prediction :", predict_spam(msg))
