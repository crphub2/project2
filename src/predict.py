import joblib
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return " ".join(filtered_tokens)

def predict_sentiment(review):
    model = joblib.load('D:/code/ravi/project/project_2/models/sentiment_model.pkl')
    vectorizer = joblib.load('D:/code/ravi/project/project_2/models/tfidf_vectorizer.pkl')

    cleaned = preprocess_text(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return "Positive" if prediction[0] == 1 else "Negative"

if __name__ == "__main__":
    while True:
        user_input = input("Enter a review (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        sentiment = predict_sentiment(user_input)
        print(f"Predicted Sentiment: {sentiment}")
