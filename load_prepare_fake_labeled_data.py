import pandas as pd
import numpy as np
from fake_review_detection_module import load_labeled_data, wordvec_features_creator, handpicked_features_creator
from preprocess_data_module import clean_text
from tqdm import tqdm 
from pathlib import Path

def load_prepare_fake_labeled_data():
    import yaml
    with open("params.yaml", "r") as file:
        params= yaml.safe_load(file)
        
    #load data
    df= load_labeled_data()
    #clean and vectorize the review_column
    print("vectorizing, lemmatizing, and removing stop words...")
    df["vectorized_reviews"]= df["review"].map(clean_text)
    feature_creator= dict(
        word2vec_features= wordvec_features_creator,
        handpicked_features= handpicked_features_creator
        )
    #getting the params with True value except pca
    active_features= [key for key, val in params["load_prepare_fake"]["features"].items() if (val == True)]
    print("selected feature_creators: ", active_features)
    #running the selected models
    for model_type in tqdm(active_features):
        print("creating {}".format(model_type))
        if feature_creator.get(model_type) == None:
            continue
        else:
            df= feature_creator.get(model_type)(df)
    #creating final feature column
    print("creating a concatenated features column...")
    df["features"]= [(np.array([vec for lst in df[active_features].values[i].flatten("C") for vec in lst])) for i in tqdm(range(len(df))) ]
    print("features column created!")
    return df

if __name__ == "__main__":
    df= load_prepare_fake_labeled_data()
    Path("data/fake/training").mkdir(parents=True, exist_ok=True)
    df.to_pickle('data/fake/training/labeled_processed.pkl')
        