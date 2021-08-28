import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
import re

def clean_categories_column(df):
    """clean_categories_column takes in a dataframe and return the categories
    per item that occur 500 or more times in the full product space

    Clean_categories_column starts by generating a total category list
    of all categories tagged to every product
    then it counts the occurrance of every cat in the list and extracts only the list 
    with 500 occurrences or more.
    finally it updates the category column to only include these categories per product to
    help with feature selection/ generation.

    Args:
        df (data_frame): data_frame that includes "category" labelled column

    Returns:
        df: same data_frame with the category column updated
    """
    #extracting the most common
    print("Extracting total category list")
    total_cat_lst= [word.lower() for lst in df["category"] for word in lst]
    print("Extracting categories with at least 500 occurrences")
    cat_counter_dict= dict(Counter(total_cat_lst))
    cat_500= [cat for cat in cat_counter_dict if cat_counter_dict[cat]>=500]
    df["category"]= df["category"].apply(lambda row : [word.lower() for word in row if word in cat_500])
    return df

def create_full_feature_column(df):
    """create_full_feature_column takes the df and combines category, description, brand, feature
    columns into one full_feature column and extract alphanumeric characters only



    Args:
        df (dataframe): target dataframe

    Returns:
        df (dataframe): updated dataframe with full_feature column
    """
    df["full_features"]= (df["category"]+ df["description"]+ df["brand"]+ df["feature"]).lower()
    df["full_features"]= df["full_features"].replace("\\t", "").replace("\\n", "")
    df["full_features"]= df["full_features"].apply(lambda row: re.findall(r'\w+', row, re.I, re.DOTALL))
    df.drop(columns= ["category", "description", "brand", "feature"], inplace= True)
    return df

if __name__ == "main)":
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument(####ADD ARGUMENTS###)
    )
    parser.add_argument(
        "output",
        help= "output path (string)")
    args= parser.parse_args()

    df= pd.read_pickle(### PATH TO DF##
                       )
    df= clean_categories_column(df)
    df= clean_vectorize_features_column(df)
    df= clean_vectorize_review_column(df)
    #saving the cleaned df
    df.to_pickle(args.output)
    