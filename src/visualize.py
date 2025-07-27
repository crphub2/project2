# visualize.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud

# Directories
os.makedirs("plots", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load cleaned data
df = pd.read_csv("data/cleaned_reviews.csv")
print(f"Data shape: {df.shape}")

# Load model and vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Sentiment label distribution
label_counts = df['label'].value_counts()

# Pie Chart
plt.figure(figsize=(6, 6))
label_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title("Sentiment Distribution (Pie Chart)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("plots/pie_chart_sentiment.png")
plt.show()

# Bar Chart
plt.figure(figsize=(6, 4))
label_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title("Sentiment Count (Bar Plot)")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/bar_chart_sentiment.png")
plt.show()

# Word Clouds
for sentiment in ['positive', 'neutral', 'negative']:
    text = " ".join(df[df['label'] == sentiment]['cleaned_review'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {sentiment.capitalize()} Reviews")
    plt.tight_layout()
    plt.savefig(f"plots/wordcloud_{sentiment}.png")
    plt.show()

# Confusion Matrix

# Drop rows with missing cleaned_review
df_pred = df.dropna(subset=['cleaned_review'])

# Predict on the cleaned dataset
X_vec = vectorizer.transform(df_pred['cleaned_review'])
y_pred = model.predict(X_vec)

# Create confusion matrix
cm = confusion_matrix(df_pred['label'], y_pred, labels=['positive', 'neutral', 'negative'])




y_true = df_pred['label']
cm = confusion_matrix(y_true, y_pred, labels=['positive', 'neutral', 'negative'])



plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['positive', 'neutral', 'negative'],
            yticklabels=['positive', 'neutral', 'negative'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Heatmap)")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.show()

# Classification Report
report = classification_report(y_true, y_pred, target_names=['positive', 'neutral', 'negative'])
print("Classification Report:\n", report)

# Save report
with open("reports/classification_report.txt", "w") as f:
    f.write(report)

print("\n✅ Visualizations and report saved in 'plots/' and 'reports/' folders.")
