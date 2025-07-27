# app.py
import streamlit as st
import joblib
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
stop_words = set(stopwords.words('english'))

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return " ".join(filtered_tokens)

# UI
st.set_page_config(page_title="Amazon Sentiment Analyzer", layout="centered")
st.title("🛒 Amazon Review Sentiment Analyzer")

review = st.text_area("Enter your product review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = preprocess_text(review)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        sentiment = "🟢 Positive" if prediction == 1 else "🔴 Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
