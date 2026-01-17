import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer

def train_and_predict(train_df, test_df):
    X = train_df['cleaned_text']
    y = train_df['Star Rating']
    
    X_test = test_df['cleaned_text']
    
    # Define Pipeline
    # Using simple TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, multi_class='multinomial'))
    ])
    
    # Internal Validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='weighted')
    
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring=f1_scorer)
    print(f"Cross-Validation Weighted F1-Scores: {scores}")
    print(f"Mean Weighted F1-Score: {np.mean(scores):.4f}")
    
    # Train on full dataset
    print("Training on full dataset...")
    pipeline.fit(X, y)
    
    # Predict
    print("Generating predictions...")
    predictions = pipeline.predict(X_test)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Star Rating': predictions
    })
    
    return submission
