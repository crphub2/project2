# Sentiment Analysis of Amazon Reviews – README

End-to-end sentiment analysis pipeline for Amazon customer reviews using Python. This project covers data loading, cleaning, exploratory analysis, sentiment scoring with VADER and TextBlob, label assignment (Positive/Negative/Neutral), class distribution visualization, and ranking reviews by helpfulness.

## Highlights

- Clear NLP workflow in a Jupyter Notebook
- Data checks: shape, dtypes, missing values, duplicates, quantiles
- Text cleaning:
  - Remove punctuation/numbers via regex
  - Lowercasing
- Sentiment engines:
  - VADER (rule/lexicon-based) for polarity scores (neg/neu/pos, compound)
  - TextBlob for polarity and subjectivity
- Final sentiment label assignment logic using VADER scores
- Visual EDA:
  - Categorical distributions (e.g., Overall ratings)
  - Sentiment distribution (bar + pie)
- Review ranking using Wilson Lower Bound to prioritize helpful reviews

## Dataset

- Amazon reviews CSV (linked in the video description)
- Typical columns observed:
  - reviewText (text)
  - overall (rating)
  - reviewerName
  - reviewTime / day_diff
  - wilson_lower_bound (precomputed helpfulness ranking metric)
  - Additional meta columns depending on the file
- Place the CSV locally and update the path in the notebook/script.

## Project Structure

- notebooks/
  - amazon_sentiment_analysis.ipynb – Full workflow (EDA, cleaning, sentiment scoring, visualization, ranking)
- src/
  - preprocessing.py – Text cleaning utilities (regex, lowercasing)
  - sentiment.py – VADER/TextBlob scoring and label assignment helpers
  - eda.py – Categorical summaries and plots
  - rank.py – Wilson lower bound sorting utilities
- data/
  - amazon_reviews.csv – Input dataset (not committed)
- reports/
  - figures/ – Generated charts (optional)

A single notebook is sufficient to follow the video; the src modules are suggested for maintainability.

## Environment Setup

- Python 3.8+
- Install dependencies:
  - pandas, numpy
  - nltk, textblob
  - matplotlib, seaborn, plotly, cufflinks
  - wordcloud
- Example:
  - pip install pandas numpy nltk textblob matplotlib seaborn plotly cufflinks wordcloud
- NLTK resources:
  - python -c "import nltk; nltk.download('vader_lexicon')"

## Workflow

1. Load data
   - df = pd.read_csv("data/amazon_reviews.csv")
   - Inspect with df.head(), df.shape, df.info()

2. Data quality checks
   - Missing analysis (isna().sum())
   - Duplicates (df.duplicated().sum())
   - Quantiles for numeric stability (df.quantile([0.05,0.5,0.95]))

3. Sorting by helpfulness
   - Sort by wilson_lower_bound descending to surface impactful reviews first

4. Text cleaning
   - Remove punctuation/numbers: re.sub("[^A-Za-z ]", " ", text)
   - Lowercase: text.lower()
   - Apply to reviewText

5. Sentiment scoring
   - VADER: SentimentIntensityAnalyzer().polarity_scores(text) → {neg, neu, pos, compound}
   - TextBlob: TextBlob(text).sentiment → polarity, subjectivity
   - Store columns: vader_neg, vader_neu, vader_pos, vader_compound, tb_polarity, tb_subjectivity

6. Sentiment labeling
   - If vader_pos > vader_neg → “Positive”
   - If vader_neg > vader_pos → “Negative”
   - Else → “Neutral”
   - Save as sentiment column

7. Visualization
   - Overall rating distribution (bar + pie)
   - Sentiment distribution (bar + pie)
   - Optional word cloud per sentiment class

8. Top-N insights
   - Positive/Negative/Neutral top 20 by wilson_lower_bound
   - Sample review texts with their sentiment scores

## Usage

From notebook:
- Open notebooks/amazon_sentiment_analysis.ipynb
- Run cells to:
  - Clean text
  - Compute VADER/TextBlob scores
  - Assign sentiment labels
  - Visualize distributions
  - Rank and display top reviews

From scripts (optional):
- python src/preprocessing.py --in data/amazon_reviews.csv --out data/clean_reviews.csv
- python src/sentiment.py --in data/clean_reviews.csv --out data/scored_reviews.csv
- python src/eda.py --in data/scored_reviews.csv --figdir reports/figures

## Example Code Snippets

Text cleaning:
```python
import re

def clean_text(t: str) -> str:
    t = re.sub(r"[^A-Za-z ]", " ", str(t))
    return t.lower().strip()
```

VADER + TextBlob scoring:
```python
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

sia = SentimentIntensityAnalyzer()

def score_sentiment(text: str):
    vs = sia.polarity_scores(text)
    tb = TextBlob(text).sentiment
    return {
        "vader_neg": vs["neg"],
        "vader_neu": vs["neu"],
        "vader_pos": vs["pos"],
        "vader_compound": vs["compound"],
        "tb_polarity": tb.polarity,
        "tb_subjectivity": tb.subjectivity,
    }

def label_from_vader(pos, neg):
    if pos > neg:
        return "Positive"
    if neg > pos:
        return "Negative"
    return "Neutral"
```

Apply to DataFrame:
```python
import pandas as pd

df = pd.read_csv("data/amazon_reviews.csv")
df["reviewText"] = df["reviewText"].map(clean_text)

scores = df["reviewText"].map(score_sentiment).apply(pd.Series)
df = pd.concat([df, scores], axis=1)
df["sentiment"] = [label_from_vader(p, n) for p, n in zip(df["vader_pos"], df["vader_neg"])]
```

## Tips

- Ensure VADER lexicon is downloaded in NLTK.
- Handle empty or NaN reviewText by filling with empty strings before scoring.
- Use handle_unknown="ignore" patterns when building categorical plots to avoid errors with unseen categories.
- Wilson Lower Bound helps rank reviews by helpfulness while accounting for uncertainty.
- For performance on large datasets, vectorize operations and avoid Python-level loops where possible.

## Extensions

- Balance classes with sampling techniques for downstream classifiers.
- Train a supervised sentiment classifier (e.g., Logistic Regression with TF-IDF).
- Leverage transformer models (e.g., DistilBERT) for improved accuracy.
- Add language detection and multilingual handling.
- Serve a simple dashboard (Streamlit) to explore sentiment interactively.

## License

Educational use. Respect the dataset’s original license and terms.
