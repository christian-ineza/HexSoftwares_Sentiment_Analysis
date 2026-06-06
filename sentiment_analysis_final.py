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
        print("[OK] Models loaded successfully!")
        return model, vectorizer
    else:
        print("[ERROR] Model files not found. Make sure both .pkl files are in the same folder.")
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

# Interactive typing feature
def interactive_mode():
    """Let user type tweets and get predictions in real-time"""
    print("\n" + "="*50)
    print("   TYPE A TWEET AND GET SENTIMENT")
    print("="*50)
    
    model, vectorizer = load_models()
    if model is None:
        return
    
    print("\nType any tweet and press Enter")
    print("Type 'exit' or 'quit' to stop")
    print("Type 'demo' to see example tweets")
    print("="*50)
    
    while True:
        print("\nYour tweet:", end=" ")
        user_input = input()
        
        if user_input.lower() in ['exit', 'quit']:
            print("\n" + "="*50)
            print("   THANK YOU FOR USING")
            print("   Hex Softwares")
            print("="*50)
            break
        
        elif user_input.lower() == 'demo':
            print("\nRunning demo tweets...\n")
            demo_tweets = [
                "I love this!",
                "This is terrible",
                "Not bad at all",
                "I have no opinion"
            ]
            for tweet in demo_tweets:
                result = predict_sentiment(tweet, model, vectorizer)
                print(f"   Tweet: {tweet}")
                print(f"   Sentiment: {result}\n")
        
        elif user_input.strip():
            result = predict_sentiment(user_input, model, vectorizer)
            print(f"\n   Tweet: {user_input}")
            print(f"   Sentiment: {result}")
            print("-"*40)

# Demo function
def run_demo():
    """Run a quick demo of the model"""
    print("\n" + "="*50)
    print("   SENTIMENT ANALYSIS DEMO")
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
    
    print("\nResults:\n")
    for tweet in test_tweets:
        result = predict_sentiment(tweet, model, vectorizer)
        print(f"   Tweet: {tweet}")
        print(f"   Sentiment: {result}\n")

if __name__ == "__main__":
    print("="*50)
    print("   SENTIMENT ANALYSIS MODEL")
    print("   Hex Softwares Internship")
    print("="*50)
    print(f"   Accuracy: 84.94%")
    print(f"   Model: Logistic Regression")
    print(f"   Ready for predictions")
    print("="*50)
    
    # Ask user what they want to do
    print("\nChoose an option:")
    print("   1. Type tweets interactively")
    print("   2. Run demo")
    print("   3. Exit")
    
    choice = input("\nEnter 1, 2, or 3: ")
    
    if choice == '1':
        interactive_mode()
    elif choice == '2':
        run_demo()
    else:
        print("\nGoodbye!")
