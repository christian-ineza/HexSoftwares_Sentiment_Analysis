"""
Sentiment Analysis Project
Hex Softwares Internship

Accuracy: 84.94%
Classes: Negative (-1), Neutral (0), Positive (1)
"""

import re
import joblib
import os
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

# Load models (automatically looks for files in same folder)
def load_models():
    """Load the trained model and vectorizer"""
    model_path = 'sentiment_model_final.pkl'
    vectorizer_path = 'vectorizer_final.pkl'
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print(" Models loaded successfully!")
        return model, vectorizer
    else:
        print(" Model files not found. Make sure both .pkl files are in the same folder.")
        return None, None

# Prediction function
def predict_sentiment(tweet, model=None, vectorizer=None):
    """Predict sentiment of a tweet"""
    if model is None or vectorizer is None:
        model, vectorizer = load_models()
        if model is None:
            return "Error: Models not loaded"
    
    cleaned = clean_text_with_negations(tweet)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    return {1: "Positive", 0: "Neutral", -1: "Negative"}[pred]

# Demo function
def run_demo():
    """Run a quick demo of the model"""
    print("\n" + "="*50)
    print("TWITTER SENTIMENT ANALYSIS DEMO")
    print("="*50)
    
    model, vectorizer = load_models()
    if model is None:
        return
    
    test_tweets = [
        "I love this!",
        "This is terrible",
        "Not bad at all",
        "I have no opinion",
        "Modi is doing great work"
    ]
    
    print("\n Results:\n")
    for tweet in test_tweets:
        result = predict_sentiment(tweet, model, vectorizer)
        print(f"Tweet: {tweet}")
        print(f"Sentiment: {result}\n")

if __name__ == "__main__":
    print("Sentiment Analysis Model Ready!")
    print("Accuracy: 84.94%")
    run_demo()
