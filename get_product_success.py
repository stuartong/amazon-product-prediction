import os
import json
import gzip
import pandas as pd
import requests
import argparse
from tqdm import tqdm
from urllib.request import urlopen

from preprocess_data import clean_categories_column, clean_text, create_full_feature_column
from preprocess_review_data import  clean_review, create_full_feature_review
from word2vec import generate_dense_features, get_pretrained_model

def get_product_success(cat_meta_url,cat_review_url):
    '''
    This function takes the meta data and review URL for a category
    and parses, cleans and converts data to Pandas dataframes.

    Inputs:
    Per-category data URL from http://deepyeti.ucsd.edu/jianmo/amazon/index.html
    cat_meta_url - URL of categories metadata
    Example: http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Video_Games.json.gz
    cat_review_url - URL of categories reviews
    Example: http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Video_Games.json.gz
    

    Outputs:
    1. combined_df - Meta data joined with success metrics from review_df
    2. review_df - Raw reviews in a dataframe
    '''
    
    # get .gz filenames to read to df
    meta_filename = cat_meta_url.split('/')[-1]
    review_filename = cat_review_url.split('/')[-1]
    
    # get meta_data and save it to file
    print("Downloading Metadata...")
    with open(meta_filename,"wb") as f:
        r = requests.get(cat_meta_url)
        f.write(r.content)
    f.close()
    print('Metadata Download Complete!')


    # get review_data and save it to file
    print("Downloading Review Data...")
    with open(review_filename,"wb") as f:
        r = requests.get(cat_review_url)
        f.write(r.content)
    f.close()
    print("Review Data Download Complete!")

    # open gzip of meta_data and get json data
    print("Reading Metadata...")
    data = []
    with gzip.open(meta_filename) as f:
        for l in tqdm(f):
            data.append(json.loads(l.strip()))
    f.close()
    
    # save meta_data to dataframe
    meta_df = pd.DataFrame.from_dict(data)
    print("Metadata Converted")

    # clean meta_data as recommended
     
    # fill NA with blanks
    meta_df.fillna('',inplace=True)
    # remove unformatted rows
    meta_df = meta_df[~meta_df.title.str.contains('getTime')]

    # open gzip of review_data and get json data
    print("Reading Review Data...")
    data = []
    with gzip.open(review_filename) as f:
        for l in tqdm(f):
            data.append(json.loads(l.strip()))
    f.close()
    
    # save review_data to dataframe
    review_df = pd.DataFrame.from_dict(data)
    
    print("Review Data Converted")
   
    # --------------------------------------------------------------------------------------------------------------------------- #

    # get success metrics - target variables
    # ONLY KEEP ONE FOR training models 
    # DO NOT USE UNUSED VARIABLE FOR TRAINING - DATA LEAK
    
    ### let us futher process the review_df alone
    #Step1 : cleaning the review_df from any duplicates in 'reviewText' column
    # review_df = clean_review_column(review_df)
    #Step2 : cleaning the text in 'reviewText', 'summary' columns
    print("cleaning ReviewText and Summary Columns started...")
    for col in ['reviewText', 'summary']:
        review_df[col]= review_df[col].apply(lambda row: clean_review(row))
    print("cleaning complete!")
    #Step3 : merging 'reviewText', 'summary' columns into 1 and extracting alphanumeric values only
    review_df = create_full_feature_review(review_df)
    #Step4 : Get and initialize pretrained word2vec model
    word2vec_model= get_pretrained_model()
    #Step5 : creating wordvec columns to the df
    review_df["wordvec"]= review_df["full_features"].apply(lambda text: generate_dense_features(tokenized_text= text, model= word2vec_model, use_mean= True))
    
    
    ### let us futher process the metadata
    summary_df = review_df.groupby('asin').agg(
        tot_stars = ('overall','sum'),
        tot_reviews = ('overall','count'),
        avg_stars = ('overall','mean')
    )
    
    # join summary with meta_data
    combined_df = meta_df.join(summary_df,on='asin')

    # fill NaN with 0 in combined_df - i.e. no reviews
    combined_df[['tot_stars','tot_reviews','avg_stars']] = combined_df[['tot_stars','tot_reviews','avg_stars']].fillna(value=0)
    
    #Step1: removing duplicate products based on "asin" column
    combined_df.drop_duplicates(subset= "asin", inplace= True)
    #Step2: downsampling the categories that only appear 500 times in column
    combined_df= clean_categories_column(combined_df)
    #step3: cleaning and vectorizing the text in 'brand', 'title', 'feature', 'category', 'description' columns
    print("extracting, filtering, and lemmatizing process initiated")
    for col in ["description", "title", "feature"]:
        combined_df[col]= combined_df[col].apply(lambda row: clean_text(row))
    #Step4:merging category, description, brand, feature columns into 1 and extracting alphanumeric values only
    combined_df= create_full_feature_column(combined_df)
    #step 5: Get and initialize pretrained word2vec model
    word2vec_model= get_pretrained_model()
    #step: creating wordvec columns to the df
    combined_df["wordvec"]= combined_df["full_features"].apply(lambda text: generate_dense_features(tokenized_text= text, model= word2vec_model, use_mean= True))
    
    
    

    # Clean Up - delete downloaded gzip files
    print("Cleaning up temp files...")
    os.remove(meta_filename)
    os.remove(review_filename)

    return combined_df, review_df, meta_filename, review_filename

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'meta_url', help='URL of Per-category Metadata (string)'
    )
    parser.add_argument(
        'review_url', help='URL of Per-category reviews (string)'
    )
    args = parser.parse_args()

    data = get_product_success(args.meta_url,args.review_url)
    data[0].to_pickle('data/combined_'+data[2][5:-8]+'.pkl')
    data[1].to_pickle('data/review_'+data[3][:-8]+".pkl")

'''
To run in terminal - use the following example command
python3.7 get_product_success.py http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Video_Games.json.gz http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Video_Games.json.gz
'''

