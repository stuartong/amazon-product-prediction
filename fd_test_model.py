import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import pickle
from sklearn import preprocessing
from fake_review_detection_module import wordvec_features_creator
from preprocess_data_module import clean_text

def detect_fake_reviews():
    with open("params.yaml", "r") as file:
        params= yaml.safe_load(file)


    
    #Importing models/vectorizers ..etc
    if params.get("fd_supervised_model").get("tfidf"):
        tfidf_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("tfidf_path")), "rb"))
    if params.get("fd_supervised_model").get("pca"):
        pca_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("pca_fitted_path")), "rb"))
    if params.get("fd_supervised_model").get("scale"):
        scaler_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("scaler_fitted_path")), "rb"))
    model_fitted= pickle.load(open(Path(params.get("fd_test_stage").get("model_fitted")), "rb"))

    #creating a dictionary of steps to run on the data
    val_dict=dict(
        tfidf= params.get("fd_supervised_model").get("tfidf"),
        word2vec_features= params.get("load_prepare_fake").get("features").get("word2vec_features")
        )
    steps_dict= dict(
        tfidf= "tfidf",
        word2vec_features= "word2vec_features",
        )

    #Import Data
    df= pd.read_pickle("data/prepared/reviews.pkl")
    df["consolidated_text"]= df["reviewText"].str.split() + df["summary"].str.split()
    print("cleaning...")
    df["vectorized_reviews"]= df["consolidated_text"].map(clean_text )
    print("cleaning complete!")
    #WORD2VEC STAGE
    if params.get("load_prepare_fake").get("features").get("word2vec_features"):
        df= wordvec_features_creator(df)
        
    #TFIDF STAGE
    if params.get("fd_supervised_model").get("tfidf"):
        tfidf_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("tfidf_path")), "rb"))
        corpus= [" ".join(lst) for lst in df["vectorized_reviews"]]
        tfidf_vec= tfidf_fitted_model.tranform(corpus)
        df["tfidf"]= [np.array(i) for i in zip(*tfidf_vec.toarray().T)]

    #conslidate stage
    
    #PCA STAGE
    if params.get("fd_supervised_model").get("pca"):
        pca_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("pca_fitted_path")), "rb"))


    #SCALE STAGE
    if params.get("fd_supervised_model").get("scale"):
        scaler_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("scaler_fitted_path")), "rb"))


    #DETECTING FAKE REVIEWS STAGE
    model_fitted= pickle.load(open(Path(params.get("fd_test_stage").get("model_fitted")), "rb"))



    #IS THIS NEEDED?!!
    category_name= params.get("fd_test_stage").get("category_name")

    

    if params.get("fd_supervised_model").get("scale"):
        print('Scaling features...')
        
    else:
        print('No scaling done')