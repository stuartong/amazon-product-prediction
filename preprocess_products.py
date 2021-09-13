import pandas as pd
import numpy as np
from preprocess_data_module import clean_categories_column, clean_text, quantify_features, consolidate_text_columns
from word2vec import get_pretrained_model, generate_dense_features
from run_pca import run_pca
import os


def preprocess_products(product_df_path):
    #import params from the params.yaml file
    import yaml
    with open("params.yaml", "r") as file:
        params= yaml.safe_load(file)
    word2vec_features= params["preprocess_products"]["word2vec_features"]
    handpicked_features= params["preprocess_products"]["handpicked_features"]
    word2vec_model_name= params["preprocess_products"]["word2vec_model_name"]
    pca= params["preprocess_products"]["pca"]
    #import df
    print("reading {} file".format(product_df_path.split('/')[-1]))
    df= pd.read_pickle(product_df_path+ "/products.pkl")    
    # fill NaN with 0 in df - i.e. no reviews
    df[['tot_stars','tot_reviews','avg_stars']] = df[['tot_stars','tot_reviews','avg_stars']].fillna(value=0)
    #Step1: removing duplicate products based on "asin" column
    df.drop_duplicates(subset= "asin", inplace= True)
    #Step2: downsampling the categories that only appear 500 times in column
    df= clean_categories_column(df)
    #step3: cleaning and vectorizing the text in 'brand', 'title', 'feature', 'category', 'description' columns
    print("extracting, filtering, and lemmatizing process initiated")
    for col in ["description", "title", "feature"]:
        df[col]= df[col].apply(lambda row: clean_text(row))
    #if else to either generat
    if handpicked_features & word2vec_features & pca:
        #quantify the number of tech1, tech2, images and length of description text for every product
        df= quantify_features(df)
        #merging category, description, brand, feature columns into 1 and extracting alphanumeric values only
        df= consolidate_text_columns(df)
        #Get and initialize pretrained word2vec model
        word2vec_model= get_pretrained_model(word2vec_model_name)
        #creating wordvec columns to the df
        df["wordvec"]= df["consolidated_text_column"].apply(lambda text: generate_dense_features(tokenized_text= text, model= word2vec_model, use_mean= True))
        df["features"]= df.apply(lambda row : np.append(row["wordvec"], row["quantified_features_array"]), axis=1)
        #turn features column into pca vectors
        df= run_pca(df)
        return df
    elif handpicked_features:
        #quantify the number of tech1, tech2, images and length of description text for every product
        df= quantify_features(df)
        df.rename(columns= {"quantified_features_array": "features"}, inplace= True)
        return df
    elif word2vec_features & pca:
        #merging category, description, brand, feature columns into 1 and extracting alphanumeric values only
        df= consolidate_text_columns(df)
        #Get and initialize pretrained word2vec model
        word2vec_model= get_pretrained_model(word2vec_model_name)
        #creating wordvec columns to the df
        df["features"]= df["consolidated_text_column"].apply(lambda text: generate_dense_features(tokenized_text= text, model= word2vec_model, use_mean= True))
        #turn features column into pca vectors
        df= run_pca(df)
        return df
    else:
        #merging category, description, brand, feature columns into 1 and extracting alphanumeric values only
        df= consolidate_text_columns(df)
        #Get and initialize pretrained word2vec model
        word2vec_model= get_pretrained_model(word2vec_model_name)
        #creating wordvec columns to the df
        df["features"]= df["consolidated_text_column"].apply(lambda text: generate_dense_features(tokenized_text= text, model= word2vec_model, use_mean= True))
        return df

if __name__ == "__main__":
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument("product_df_path", help= "path to target dataframe location (str)")
    args= parser.parse_args()
    #running the function
    df= preprocess_products(args.product_df_path)
    
    if os.path.isdir("data"):
        if os.path.isdir("data/products"):
            df.to_pickle('data/products/products.pkl')
        else:
            parent_dir= "data"
            directory= "products"
            path= os.path.join(parent_dir, directory)
            os.makedirs(path, exist_ok= True)
            df.to_pickle('data/products/products.pkl')
    else:
        parent_dir= os.mkdir("data")
        directory= "products"
        path= os.path.join(parent_dir, directory)
        os.makedirs(path)
        df.to_pickle('data/products/products.pkl')   
    
    
    




