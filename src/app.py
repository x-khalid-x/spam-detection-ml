import streamlit as st
import pickle
from preprocess import clean_text

# ===============================
# Charger modèle & vectorizer
# ===============================
with open("models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ===============================
# Initialiser historique si non existant
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# Interface
# ===============================
st.title("📩 Spam Detection (English Only)")
st.write("Enter one or multiple messages (one per line).")

text = st.text_area("Messages")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter at least one message.")
    else:
        messages = [m.strip() for m in text.split("\n") if m.strip()]
        cleaned = [clean_text(m) for m in messages]

        vect = vectorizer.transform(cleaned)

        # Probabilités
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vect)
        else:
            from scipy.special import expit
            probs = expit(model.decision_function(vect))
            probs = [[1 - p, p] for p in probs]

        preds = model.predict(vect)

        # Ajouter au history
        for i, msg in enumerate(messages):
            st.session_state.history.append({
                "message": msg,
                "prediction": preds[i],
                "probability": probs[i][list(model.classes_).index("spam")]
            })

# ===============================
# Affichage de l'historique
# ===============================
if st.session_state.history:
    st.subheader("📝 History of analyzed messages")
    for item in reversed(st.session_state.history):  # afficher le plus récent en premier
        msg = item["message"]
        pred = item["prediction"]
        prob = item["probability"]
        if pred == "spam":
            st.error(f"🚫 {msg} (SPAM, Prob: {prob:.2f})")
        else:
            st.success(f"✅ {msg} (NON-SPAM, Prob: {prob:.2f})")

