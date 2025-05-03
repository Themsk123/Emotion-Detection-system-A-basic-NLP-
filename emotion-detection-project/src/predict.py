import joblib
from preprocessing import clean_text
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'emotion_model.pkl'
VECTORIZER_PATH = BASE_DIR / 'models' / 'vectorizer.pkl'

# Map numeric labels to emotion names
EMOTION_MAP = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise"
}

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_emotion(text):
    clean = clean_text(text)
    if not clean or not clean.strip():
        return "neutral"
    X = vectorizer.transform([clean])
    pred = model.predict(X)
    # Map the predicted number to the emotion name
    return EMOTION_MAP.get(pred[0], "Unknown")

if __name__ == "__main__":
    # Example interactive prediction
    while True:
        user_input = input("Enter a sentence (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        print("Predicted emotion:", predict_emotion(user_input))
