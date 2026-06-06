# HexSoftwares_Sentiment_Analysis

## Performance
- **Accuracy**: 84.94%
- **Algorithm**: Logistic Regression
- **Classes**: Positive, Neutral, Negative

## Example
| Tweet | Sentiment |
|-------|-----------|
| "I love this!" | Positive |
| "This is terrible" | Negative |
| "Not bad" | Positive |

## Files
- `sentiment_model_final.pkl` - Trained model
- `vectorizer_final.pkl` - TF-IDF vectorizer  
- `sentiment_analysis_final.py` - Complete code

## How to Run

### Option 1: Google Colab (Recommended)
1. Upload `sentiment_model_final.pkl`, `vectorizer_final.pkl`, and `sentiment_analysis_final.py` to Colab
2. Run the Python file

### Option 2: Local Machine
```bash
pip install scikit-learn joblib pandas numpy
python sentiment_analysis_final.py
