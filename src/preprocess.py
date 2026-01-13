import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """Nettoyage avancé d’un texte : minuscules, ponctuation, chiffres, stopwords, stemming"""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', '', text)
    text = text.strip()
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)
