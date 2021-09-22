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
    review_path = Path("data/reviews/reviews.pkl")
    # load fake review prediction labels
    with open(r"data/fake/fake_free_data/fake_free_reviews.npy","rb") as file:
        predictions = np.load(file)

    #importing product_df, review_df
    product_df, review_df= [pd.read_pickle(file) for file in [product_path,  review_path ]]
    # filter review_df if fake_free_data == True
    if fake_free_data:
        print(f"identified {sum(predictions==0)} fake reviews")
        print("filtering the fake reviews")
        pred = pd.Series(predictions)
        review_df = review_df.filter(items= pred[pred==1].index, axis=0)
    else:
        print("Fake reviews not removed")
    product_df= create_success_metric(product_df, review_df, cutoff= cutoff)
    # fill NaN with 0 in df - i.e. no reviews
    product_df[['tot_stars','tot_reviews','avg_stars']] = product_df[['tot_stars','tot_reviews','avg_stars']].fillna(value=0)
    check_score_rank(product_df)
    return product_df

if __name__ == "__main__":
    product_df= create_success_metrics()

    Path("data/metrics").mkdir(parents= True, exist_ok= True)
    product_df.to_pickle('data/metrics/products.pkl') 
    