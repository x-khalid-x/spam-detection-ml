import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# 📌 Forcer le téléchargement dans Streamlit Cloud
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)
