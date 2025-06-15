import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('GRU-EmotionDetector.keras')
tokenizer = pickle.load(open('GRU-tokenizer.pkl', 'rb'))
labelencoder = pickle.load(open('GRU-labelencoder.pkl', 'rb'))

max_len = 100

st.title("Emotion Detector - GRU MODEL")
st.write("Enter a comment to get its emotion")
user_input = st.text_input("Your sentence:")

def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padding = pad_sequences(seq, maxlen=max_len) 
    pred = model.predict(padding)
    emotion = labelencoder.inverse_transform([np.argmax(pred)])
    return emotion[0], pred[0][np.argmax(pred)]

if st.button("Predict Emotion"):
    if user_input:
       emotion, confidence = predict_emotion(user_input)
       st.markdown(f"### Predicted Emotion: **{emotion}**")
       st.markdown(f"Confidence: `{confidence:.2f}`")
else:
    st.warning("Enter comment for prediction")