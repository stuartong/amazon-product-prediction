import pandas as pd
import numpy as np
from preprocess_data_module import clean_review_column, clean_review, create_full_feature_review

def preprocess_reviews(review_df_path):
    
    review_df= pd.read_pickle(review_df_path)
    #Step1 : cleaning the review_df from any duplicates in 'reviewText' column
    review_df = clean_review_column(review_df)
    #Step2 : cleaning the text in 'reviewText', 'summary' columns
    for col in ['reviewText', 'summary']:
        review_df[col]= review_df[col].apply(lambda row: clean_review(row))
    #Step3 : merging 'reviewText', 'summary' columns into 1 and extracting alphanumeric values only
    # review_df = create_full_feature_review(review_df)