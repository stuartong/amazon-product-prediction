from nltk import parse
import pandas as pd
import numpy as np
from pandas.io import parsers
from preprocess_data_module import clean_review_column, clean_review, create_full_feature_review
import os

def preprocess_reviews(review_df_path):

    #importing data
    print("reading {} file".format(review_df_path.split('/')[-1]))
    review_df= pd.read_pickle(review_df_path+ "/reviews.pkl")  
    
    #Step1 : cleaning the review_df from any duplicates in 'reviewText' column
    review_df = clean_review_column(review_df)
    #Step2 : cleaning the text in 'reviewText', 'summary' columns
    for col in ['reviewText', 'summary']:
        review_df[col]= review_df[col].apply(lambda row: clean_review(row))
        
    ###NOTE_TO_SELF: NEED TO FIX THIS PART BELOW-- CHECK DATA TYPES TO CONCAT###
    #Step3 : merging 'reviewText', 'summary' columns into 1 and extracting alphanumeric values only
    review_df = create_full_feature_review(review_df)
    return review_df

if __name__ == "__main__":
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument("review_df_path", help = "path to review data frame (str)")
    args= parser.parse_args()
    review_df= preprocess_reviews(args.review_df_path)
    if os.path.isdir("data"):
        if os.path.isdir("data/features"):
            review_df.to_pickle('data/features/reviews.pkl')
        else:
            parent_dir= "data"
            directory= "features"
            path= os.path.join(parent_dir, directory)
            os.makedirs(path, exist_ok= True)
            review_df.to_pickle('data/features/reviews.pkl')
    else:
        parent_dir= os.mkdir("data")
        directory= "features"
        path= os.path.join(parent_dir, directory)
        os.makedirs(path)
        review_df.to_pickle('data/features/reviews.pkl') 