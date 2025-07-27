import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only once)
nltk.download("stopwords")

# Read CSV file (not Excel)
df = pd.read_csv("data/reviews.csv")

# Show column names to verify
print("Column Names:", df.columns.tolist())

# Define English stopwords
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()  # lowercase
    text = re.sub(r"http\\S+", "", text)  # remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.strip()  # remove extra spaces
    tokens = text.split()  # tokenize
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return " ".join(tokens)

# Apply cleaning to reviewText column
df['cleaned_review'] = df['reviewText'].apply(clean_text)


# Function to assign sentiment label based on 'overall' rating
def assign_sentiment_label(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

# Apply to DataFrame
df['label'] = df['overall'].apply(assign_sentiment_label)


# Save cleaned results to CSV
df.to_csv("data/cleaned_reviews.csv", index=False)

print("✅ Preprocessing complete. Cleaned data saved to 'data/cleaned_reviews.csv'")
