import os
import sys
import pandas as pd
from preprocess import load_data, preprocess_data
from model import train_and_predict

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, 'train.csv')
    test_path = os.path.join(base_dir, 'test.csv')
    output_path = os.path.join(base_dir, 'folder_assignment', 'predictions.csv')
    
    print(f"Loading data from {train_path} and {test_path}...")
    try:
        train_df, test_df = load_data(train_path, test_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Preprocessing data...")
    train_df = preprocess_data(train_df, is_train=True)
    test_df = preprocess_data(test_df, is_train=False)
    
    print("Training model and predicting...")
    submission_df = train_and_predict(train_df, test_df)
    
    print(f"Saving predictions to {output_path}...")
    submission_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
