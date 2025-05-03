import streamlit as st
from src.predict import predict_emotion

st.title("Customer Support Emotion Detector")

user_input = st.text_area("Enter customer chat message:")

if st.button("Detect Emotion"):
    if user_input.strip():
        emotion = predict_emotion(user_input)
        st.success(f"Predicted Emotion: {emotion}")
    else:
        st.warning("Please enter a message.")
