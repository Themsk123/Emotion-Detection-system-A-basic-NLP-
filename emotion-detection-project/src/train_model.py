import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
from preprocessing import clean_text

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'training.csv'
MODEL_DIR = BASE_DIR / 'models'

def load_and_clean_data():
    df = pd.read_csv(
        DATA_PATH,
        delimiter=',',
        names=['text', 'emotion'],
        header=0,  # Skip header row if present
        dtype={'text': str}
    )
    
    # Convert emotion to numeric, handle errors
    df['emotion'] = pd.to_numeric(df['emotion'], errors='coerce')
    df = df.dropna(subset=['text', 'emotion'])
    df = df[df['text'].str.strip().astype(bool)]
    
    # Clean text
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.strip().astype(bool)]
    
    return df

def train_and_save_model():
    df = load_and_clean_data()
    
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', ngram_range=(1,2))
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['emotion'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / 'emotion_model.pkl')
    joblib.dump(vectorizer, MODEL_DIR / 'vectorizer.pkl')
    
    print("Training successful!")

if __name__ == "__main__":
    train_and_save_model()
