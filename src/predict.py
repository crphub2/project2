import joblib

def predict_sentiment(text):
    model = joblib.load('data/sentiment_model.pkl')
    return model.predict([text])[0]

if __name__ == "__main__":
    while True:
        user_input = input("Enter a review (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        sentiment = predict_sentiment(user_input)
        print(f"Predicted Sentiment: {sentiment}\n")
