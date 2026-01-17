# Google Play Store App Rating Prediction

## Problem Overview
The goal is to predict the star rating (1-5) of an app based on user reviews and other attributes. This is a multi-class classification problem.

## Approach
1.  **Data Preprocessing**:
    -   Combine `Review Title` and `Review Text` into a single text column.
    -   Handle missing values by filling them with empty strings.
    -   Text cleaning: Lowercasing, removing non-alphabetic characters.
2.  **Feature Extraction**:
    -   Use `TfidfVectorizer` to convert text data into numerical features.
3.  **Modeling**:
    -   Use `LogisticRegression` (with balanced class weights) as the baseline model. It performs well on high-dimensional sparse data (text).
4.  **Validation**:
    -   Use Stratified K-Fold Cross-Validation to evaluate the model using Weighted F1-Score.

## Instructions to Run the Code
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the pipeline:
    ```bash
    python src/pipeline.py
    ```
    This will generate `predictions.csv` in the `folder_assignment` directory.

## Directory Structure
-   `src/`: Contains source code.
-   `predictions.csv`: Generated predictions file.
-   `requirements.txt`: List of dependencies.
