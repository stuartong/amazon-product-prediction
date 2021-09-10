import os
import json
import gzip
import pandas as pd
import requests
import argparse
from tqdm import tqdm
from urllib.request import urlopen


def load_data(cat_meta_url,cat_review_url):
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

    ### let us futher process the metadata
    summary_df = review_df.groupby('asin').agg(
        tot_stars = ('overall','sum'),
        tot_reviews = ('overall','count'),
        avg_stars = ('overall','mean')
    )
    
    # join summary with meta_data
    combined_df = meta_df.join(summary_df,on='asin')

  
    
    
    

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

    product_df, review_df, meta_filename, review_filename = load_data(args.meta_url,args.review_url)
    # data[0].to_pickle('data/combined_'+data[2][5:-8]+'.pkl')
    # data[1].to_pickle('data/review_'+data[3][:-8]+".pkl")

'''
To run in terminal - use the following example command
python3.7 load_data.py http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles2/meta_Video_Games.json.gz http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Video_Games.json.gz
'''

