import pandas as pd
import numpy as np
from preprocess_data_module import clean_categories_column, clean_text,  consolidate_text_columns, wordvec_features_creator, handpicked_features_creator
from tqdm import tqdm
from pathlib import Path 


def preprocess_products(product_df_path):
    #import params from the params.yaml file
    import yaml
    with open("params.yaml", "r") as file:
        params= yaml.safe_load(file)
    
    #import df
    print("reading {} file".format(product_df_path.split('/')[-1]))
    product_df_path= Path(product_df_path)
    df= pd.read_pickle(product_df_path /  "products.pkl")    
    
    #Step1: removing duplicate products based on "asin" column
    df.drop_duplicates(subset= "asin", inplace= True)
    #Step2: downsampling the categories that only appear 500 times in column
    df= clean_categories_column(df)
    #step3: cleaning and vectorizing the text in 'brand', 'title', 'feature', 'category', 'description' columns
    print("extracting, filtering, and lemmatizing process initiated")
    for col in ["description", "title", "feature"]:
        df[col]= df[col].apply(lambda row: clean_text(row))
    #quantify the number of tech1, tech2, images and length of description text for every product
    df= consolidate_text_columns(df)

    #creating a dictionary with the possible feature_creator functions
    feature_creator= dict(
    word2vec_features= wordvec_features_creator,
    handpicked_features= handpicked_features_creator
    )   
    #getting the params with True value except pca
    active_features= [key for key, val in params["preprocess_products"].items() if (val == True) & (key != "pca") & (key!= "tfidf")]
    print("selected feature_creators: ", active_features)
    #running the selected models
    for model_type in tqdm(active_features):
        print("creating {}".format(model_type))
        if feature_creator.get(model_type) == None:
            continue
        else:
            df= feature_creator.get(model_type)(df)
    #concatenating arrays from every selected feature into one
    print("creating a concatenated features column...")
    df["features"]= [(np.array([vec for lst in df[active_features].values[i].flatten("C") for vec in lst])) for i in tqdm(range(len(df))) ]
    print("features column created!")
    

    return df

if __name__ == "__main__":
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument("product_df_path", help= "path to target dataframe location (str)")
    args= parser.parse_args()
    #running the function
    df= preprocess_products(args.product_df_path)
    Path("data/products").mkdir(parents=True, exist_ok=True)
    df.to_pickle('data/products/products.pkl')