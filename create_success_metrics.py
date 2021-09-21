import pandas as pd 
import numpy as np
from create_success_metric_module import create_success_metric, check_score_rank
import os
from pathlib import Path

def create_success_metrics():
    import yaml
    with open("params.yaml", "r") as file:
        params= yaml.safe_load(file)
    cutoff= params["success_metrics"]["cutoff"]
    fake_free_data= params["success_metrics"]["fake_free_data"]
    product_path= Path("data/products/products.pkl")
    review_path= Path("data/fake/fake_free_data/fake_free_reviews.pkl") if fake_free_data else Path("data/reviews/reviews.pkl")
    #importing product_df, review_df
    product_df, review_df= [pd.read_pickle(file) for file in [product_path,  review_path / "reviews.pkl"]]
    product_df= create_success_metric(product_df, review_df, cutoff= cutoff)
    # fill NaN with 0 in df - i.e. no reviews
    product_df[['tot_stars','tot_reviews','avg_stars']] = product_df[['tot_stars','tot_reviews','avg_stars']].fillna(value=0)
    check_score_rank(product_df)
    return product_df

if __name__ == "__main__":
    product_df= create_success_metrics()

    Path("data/metrics").mkdir(parents= True, exist_ok= True)
    product_df.to_pickle('data/metrics/products.pkl') 
    