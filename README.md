# Emotion Detection NLP Project

This project uses Natural Language Processing (NLP) and Machine Learning to detect emotions in text sentences. It is designed to train on labeled data and predict the emotion expressed in any new sentence.

---

## 🚀 Features

- **Text Preprocessing:** Cleans, tokenizes, and lemmatizes text, preserving important emotional cues (like negations).
- **Feature Extraction:** Uses TF-IDF vectorization (with optional n-grams) to convert text into numerical features.
- **Emotion Classification:** Trains a machine learning model to classify emotions.
- **User Prediction:** Predicts the emotion of any input sentence via command line or Python function.
- **Extensible:** Modular code for easy upgrades (e.g., new models, more data, web app integration).

---

## Emotion Labels

| Label | Emotion   |
|-------|-----------|
| 0     | Sadness   |
| 1     | Joy       |
| 2     | Love      |
| 3     | Anger     |
| 4     | Fear      |
| 5     | Surprise  |

---

## 📁 Project Structure
emotion-detection-project/
├── data/
│ └── raw/
│ └── training.csv
├── models/
│ ├── emotion_model.pkl
│ └── vectorizer.pkl
├── src/
│ ├── preprocessing.py
│ ├── train_model.py
│ └── predict.py
└── README.md

---

## 🛠️ Getting Started

### 1. **Install Requirements**
pip install pandas scikit-learn nltk joblib

### 2. **Prepare Data**

- Put your labeled CSV (`training.csv`) in `data/raw/`.
- Format:

text,emotion
I feel happy,1
I am so sad,0
...

### 3. **Train the Model**
python -m src.train_model

- This will preprocess the data, train the model, and save it to the `models/` directory.

### 4. **Predict Emotion**

#### **Interactive Prediction**

python -m src.predict

- Enter a sentence and get the predicted emotion.

#### **From Python Code**

from src.predict import predict_emotion

print(predict_emotion("I am thrilled to see you!")) # Output: Joy


---

## 📝 Customization & Tips

- **Improve Accuracy:** Add more labeled data, tune preprocessing, or try advanced models (like BERT).
- **Web App:** Integrate with Streamlit or Flask for a web interface.
- **Label Mapping:** Adjust `EMOTION_MAP` in `predict.py` if your label mapping is different.

---

## ❓ Troubleshooting

- **Empty vocabulary error:** Check your CSV delimiter and text cleaning logic.
- **Model predicts numbers:** Update `predict.py` to map numbers to emotion names.

---

## 📚 License

This project is for educational and research use.

---

**Enjoy detecting emotions in text! If you have questions or want to extend the project, feel free to open an issue or pull request.**

