import pandas as pd 
import numpy as np
from create_success_metric_module import create_success_metric, check_score_rank
import os

def create_success_metrics(product_path, review_path):
    import yaml
    with open("params.yaml", "r") as file:
        params= yaml.safe_load(file)
    cutoff= params["success_metrics"]["cutoff"]
    #importing product_df, review_df
    product_df, review_df= [pd.read_pickle(file) for file in [product_path + "/products.pkl",  review_path + "/reviews.pkl"]]
    product_df= create_success_metric(product_df, review_df, cutoff= cutoff)
    # fill NaN with 0 in df - i.e. no reviews
    product_df[['tot_stars','tot_reviews','avg_stars']] = product_df[['tot_stars','tot_reviews','avg_stars']].fillna(value=0)
    check_score_rank(product_df)
    return product_df

if __name__ == "__main__":
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument("products_path", help= "path to the target product data frame (str)")
    parser.add_argument("reviews_path", help= "path to the target review data frame (str)")
    # parser.add_argument("cutoff", help= "cuttoff mark of number of points to consider a product good or bad")
    args= parser.parse_args()
    product_df= create_success_metrics(args.products_path, args.reviews_path)
    if os.path.isdir("data"):
        if os.path.isdir("data/metrics"):
            product_df.to_pickle('data/metrics/products.pkl')
        else:
            parent_dir= "data"
            directory= "metrics"
            path= os.path.join(parent_dir, directory)
            os.makedirs(path, exist_ok= True)
            product_df.to_pickle('data/metrics/products.pkl')
    else:
        parent_dir= os.mkdir("data")
        directory= "metrics"
        path= os.path.join(parent_dir, directory)
        os.makedirs(path)
        product_df.to_pickle('data/metrics/products.pkl') 
    