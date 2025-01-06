import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import time

# Set page config
st.set_page_config(page_title="Sentiment Analysis", page_icon="😊", layout="wide")

# Custom CSS for animations, cursor, and enhanced styling
st.markdown("""
<style>
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
@keyframes fadeIn {
    from {opacity: 0; transform: scale(0.95);}
    to {opacity: 1; transform: scale(1);}
}
body {
    cursor: url('https://cdn.custom-cursor.com/cursors/animated-click-pointer.cur'), auto;
}
.stApp {
    background: linear-gradient(-45deg, #1e3c72, #2a5298, #23d5ab, #23a6d5);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
    color: #ffffff;
    font-family: 'Arial', sans-serif;
}
.stTextInput>div>div>input, .stTextArea textarea {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 8px;
    font-size: 1rem;
    color: #333;
}
.stButton button {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    color: #fff;
    border: none;
    padding: 10px 20px;
    border-radius: 25px;
    font-size: 1rem;
    cursor: pointer;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.stButton button:hover {
    transform: scale(1.1);
    box-shadow: 0 4px 15px rgba(30, 60, 114, 0.6);
}
.sentiment-result {
    padding: 20px;
    border-radius: 10px;
    animation: fadeIn 0.8s ease-out;
    font-size: 1.2rem;
    color: #333;
    text-align: center;
}
h2 {
    font-family: 'Georgia', serif;
}
p {
    font-family: 'Verdana', sans-serif;
}
.example-button button {
    background: linear-gradient(to right, #ff7e5f, #feb47b);
    color: #fff;
    border: none;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.example-button button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 10px rgba(255, 126, 95, 0.5);
}
</style>
""", unsafe_allow_html=True)

# Constants
MAX_LEN = 200

@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('sentiment_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def preprocess_text(text, tokenizer):
    words = text.lower().split()
    return [tokenizer.get(word, 2) for word in words]  # 2 is the index for unknown words

def predict_sentiment(text, model, tokenizer):
    sequence = preprocess_text(text, tokenizer)
    padded = pad_sequences([sequence], maxlen=MAX_LEN)
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative", prediction

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Streamlit app
st.title("🎭 Advanced Sentiment Analysis")

st.write("""
This app uses a deep learning model trained on the IMDB movie review dataset to predict the sentiment of your text.
Enter your text below and click 'Analyze' to see the result!
""")

# Text input
text_input = st.text_area("Enter your text here:", height=100)

# Analyze button
if st.button("Analyze"):
    if text_input:
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = predict_sentiment(text_input, model, tokenizer)
            time.sleep(1)  # Add a small delay for effect
        
        if sentiment == "Positive":
            st.markdown(f"""
            <div class="sentiment-result" style="background-color: rgba(0, 255, 0, 0.2);">
                <h2>😃 Positive Sentiment</h2>
                <p>The model predicts a positive sentiment with {confidence:.2f} confidence.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="sentiment-result" style="background-color: rgba(255, 0, 0, 0.2);">
                <h2>😞 Negative Sentiment</h2>
                <p>The model predicts a negative sentiment with {1-confidence:.2f} confidence.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.")

# Add some example texts for users to try
st.subheader("Try these examples:")
examples = [
    "This movie was absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat.",
    "I was really disappointed with this film. The characters were poorly developed and the story was predictable.",
    "While the special effects were impressive, the dialogue felt forced and unnatural. Overall, a mediocre experience."
]
for i, example in enumerate(examples):
    if st.button(f"Example {i+1}", key=f"example_btn_{i}", help="Click to analyze this example", use_container_width=True):
        st.text_area("Text input", value=example, height=100, key=f"example_{i}")
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence = predict_sentiment(example, model, tokenizer)
            time.sleep(1)  # Add a small delay for effect
        
        if sentiment == "Positive":
            st.markdown(f"""
            <div class="sentiment-result" style="background-color: rgba(0, 255, 0, 0.2);">
                <h2>😃 Positive Sentiment</h2>
                <p>The model predicts a positive sentiment with {confidence:.2f} confidence.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="sentiment-result" style="background-color: rgba(255, 0, 0, 0.2);">
                <h2>😞 Negative Sentiment</h2>
                <p>The model predicts a negative sentiment with {1-confidence:.2f} confidence.</p>
            </div>
            """, unsafe_allow_html=True)