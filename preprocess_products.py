import pandas as pd
import numpy as np
from preprocess_data_module import clean_categories_column, clean_text, quantify_features, consolidate_text_columns
from word2vec import get_pretrained_model, generate_dense_features


def preprocess_products(product_df_path, word2vec_features= True, handpicked_features= True, word2vec_model_name= 'glove-wiki-gigaword-300'):
    #import df
    print("reading {} file".format(product_df_path.split('/')[-1]))
    df= pd.read_pickle(product_df_path)    
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
    if handpicked_features & word2vec_features:
        #quantify the number of tech1, tech2, images and length of description text for every product
        df= quantify_features(df)
        #merging category, description, brand, feature columns into 1 and extracting alphanumeric values only
        df= consolidate_text_columns(df)
        #Get and initialize pretrained word2vec model
        word2vec_model= get_pretrained_model(word2vec_model_name)
        #creating wordvec columns to the df
        df["wordvec"]= df["consolidated_text_column"].apply(lambda text: generate_dense_features(tokenized_text= text, model= word2vec_model, use_mean= True))
        df["features"]= df.apply(lambda row : np.append(row["wordvec"], row["quantified_features_array"]), axis=1)
        return df
    elif handpicked_features:
        #quantify the number of tech1, tech2, images and length of description text for every product
        df= quantify_features(df)
        df.rename(columns= {"quantified_features_array": "features"}, inplace= True)
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
    parser.add_argument("word2vec_features", help= "Boolean to create word2vec feature array")
    parser.add_argument("handpicked_features", help= "Boolean to create hand_picked feature representation of tech1, tech2, number of images, length of words in description space")
    parser.add_argument("word2vec_model_name", help= "word2vec model name")
    args= parser.parse_args()
    #running the function
    df= preprocess_products(args.product_df_path, args.word2vec_features, args.handpicked_features, args.word2vec_model_name)
    df.to_pickle(args.product_df_path+".pkl")
    
    




