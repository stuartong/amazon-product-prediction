# Amazon Product Success Prediction

## Overview
The purpose of this repo is to consolidate and track code related to our Milestone 2 Project.

We plan to accomplish 3 main things:
1. Predict the likelihood of success of a product (defined by a metric related to total/average number of stars or salesrank) from product metadata as features
2. Feature Engineering - Use topic modeling and NLP techniques to create new target variables that represent keywords/sentiment from review text as new target variables to see if we can predict and enable a merchant to understand how buyers on the platform would react to a newly launch prodcut with similar features
3. Clustering - We intend to use clustering to improve product search enabling customer to search for products with similar features and potentially use methods like DBScan for outlier detection to see if we can pick up fake reviews

## Key Links
- [Full Proposal](https://docs.google.com/document/d/1qO5qy0LVd6yYzDum-VQhvQ-wNo1fDjyhly71D3Y5NtQ/edit)
- [Data Set](http://deepyeti.ucsd.edu/jianmo/amazon/index.html)
- [Project Tracker](https://docs.google.com/spreadsheets/d/1cw7917PWv5VBahoYk_9mUvYMjN5BM_jAM6zCysu4KYE/edit#gid=0)

## Pipeline

### 1. get_product_success.py
For a specific category, summarize reviews to create product success metric and join on category metadata table to form the full dataset for that category, include:

1. Data Exploration
2. Sanity checks - duplicates/NANs etc
3. Data Summary

Input: review_df, metadata_df

Output: combined_df


### 2. clean_text.py
Clean and tokenize product reviews and descriptions. Split string on spaces and carries out the following steps:

1. punctuation removal
2. lowercase all words
3. Tokenize 
4. Lemmatize 
5. Remove stopwords

Input: Product description/review string

Output: List of tokens 

### 3. vectorize_XXX.py
Convert tokens to vectorzied format using method XXX. Note might need multiple vectorizing functions for different purposes.

Input: List or df_column

Output: Updated list or df_column

### 4. run_supervised.py
Takes a cleaned dataframe as input with Xs and Y specified. High level steps as follows:

1. Vectorize (if required)
2. Split to train/test
3. Train pre-determined model/models
4. Train dummy & baseline models
5. Run predictions on test data
6. Evalulate Models
7. Create evaluation tables/visuals (if any)

Input: cleaned dataframe

Output: Trained model, Evaluation details, Visuals

### 5. create_features.py
Takes a cleaned dataframe as input and uses topic modeling methods to extract features from product reivews and descriptions. Creates new features to be used as Xs and Y in run_supervised.py that are appended to the dataframe that can be used in run_supervised.py.

Input: cleaned dataframe

Output: updated dataframe with new features

### 6. clustering_features.py
Clustering on updated dataframe with new features to to find products that are similar.

Input: Updated dataframe with new features

Output: KNN decision boundary plot, dendograms, t-SNE

### 7. clustering_reviews.py
Clean, tokenenize and vectorize a specific category/ multiple categories review data and runs clustering to identify potential outliers interesting cluster that could signal fake reviews. 

Input: review_df

Output: Cluster plot

### 8. clustering_product_reviews.py
Uses combined_df to cluster products to find outliers that could signal potential fake reviews or interesting structure.

Input: combined_df

Output: Cluster plot