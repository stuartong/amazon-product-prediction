import pandas as pd 
import numpy as np
from create_success_metric_module import create_success_metric, check_score_rank

def create_success_metrics(product_df_path, review_df_path, cutoff):
    #importing product_df, review_df
    product_df, review_df= [pd.read_pickle(path) for path in [product_df_path, review_df_path]]
    product_df= create_success_metric(product_df, review_df, cutoff= cutoff)
    check_score_rank(product_df)
    return product_df

if __name__ == "__main__":
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument("product_df_path", help= "path to the target product data frame (str)")
    parser.add_argument("review_df_path", help= "path to the target review data frame (str)")
    parser.add_argument("cutoff", help= "cuttoff mark of number of points to consider a product good or bad")
    args= parser.parse_args()
    product_df= create_success_metrics(args.product_df_path, args.review_df_path, args.cutoff)
    product_df.to_pickle(args.product_df_path + "pkl")