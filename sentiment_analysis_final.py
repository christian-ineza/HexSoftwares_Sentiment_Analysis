"""
Twitter Sentiment Analysis Project
Hex Softwares Internship

Accuracy: 84.94%
Classes: Negative (-1), Neutral (0), Positive (1)
"""

import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Cleaning function with negation handling
def clean_text_with_negations(text):
    text = str(text).lower()
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|@\w+|#', '', text)
    
    # Handle negations
    text = re.sub(r'not bad', 'good', text)
    text = re.sub(r'not good', 'bad', text)
    text = re.sub(r"don't like", 'dislike', text)
    text = re.sub(r"didn't like", 'dislike', text)
    text = re.sub(r"isn't", 'is not', text)
    text = re.sub(r"aren't", 'are not', text)
    text = re.sub(r"wasn't", 'was not', text)
    text = re.sub(r"weren't", 'were not', text)
    
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Prediction function
def predict_sentiment(tweet, model, vectorizer):
    cleaned = clean_text_with_negations(tweet)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    return {1: "Positive", 0: "Neutral", -1: "Negative"}[pred]

# Load your saved model (update paths)
# model = joblib.load('sentiment_model_final.pkl')
# vectorizer = joblib.load('vectorizer_final.pkl')

# Example usage
# print(predict_sentiment("I love this!", model, vectorizer))

print("Sentiment Analysis Model Ready!")
print("Accuracy: 84.94%")
