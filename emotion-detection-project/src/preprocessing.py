import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Keep negations for emotion detection
NEGATIONS = {"not", "no", "nor", "never", "don't", "won't", "didn't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't"}
CUSTOM_STOPWORDS = set(stopwords.words('english')) - NEGATIONS

def clean_text(text):
    text = str(text).lower()
    # Keep apostrophes for contractions
    text = re.sub(r"[^a-zA-Z\s']", '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens) if tokens else None
