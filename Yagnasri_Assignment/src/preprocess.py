import pandas as pd
import re
import numpy as np

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def preprocess_data(df, is_train=True):
    # Fill missing values
    df['Review Text'] = df['Review Text'].fillna('')
    df['Review Title'] = df['Review Title'].fillna('')
    
    # Combine title and text
    df['full_text'] = df['Review Title'] + " " + df['Review Text']
    
    # Clean text
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    return df
