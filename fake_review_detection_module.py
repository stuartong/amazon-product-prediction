#This file acts as a reference library to the fake_review.py file
import pandas as pd
import numpy as np
import os
import csv
import yaml

def load_labeled_data():
    """load_labeled_data takes in a file path from the params.yaml and returns a df

    dataframe includes Rating, Verified, Category, Review, and Label columns
    Rating [int]: Is the rating given to a each review
    Verified [binary]: Whether is user is verified or not (1 for Verified, 0 for Not Verified)
    Category [str]: Name of product main parent category
    Review [str]: Review given to the product
    label [binary]: Whether the review is a fake or real (0 for fake, 1 for real)
    
    Returns:
        [type]: [description]
    """
    
    with open("params.yaml", "r") as file:
        params= yaml.safe_load(file)
    
    labeled_file_path= params["fake_review_labeled_file_path"]
    #reading the data
    data= []
    with open(labeled_file_path, encoding= "utf8") as file:
        reader= csv.reader(file, delimiter= "\t")
        next(reader)
        for line in reader:
            data.append((line[1],line[2], line[3], line[4], line[8]))
    df= pd.DataFrame(data, columns=["labeltag", "rating", "verified", "category", "review"] )
    df["label"]= np.where(df["labeltag"]== "__label1__", 0, 1)
    df["verified"]= np.where(df["verified"]== "Y", 1, 0)
    df["category"]= df["category"].astype("category")
    df.drop(["labeltag"], axis=1, inplace= True)
    return df

def stratified_split(file_path):
    """stratified_split creates a train, test split that maintains the distribution of labels in the data
    Args:
        file_path (str): path of the location of the target data frame
    Returns:
        X_train, X_test, y_train, y_test: numpy arrays of train, test data ready for next steps
    """

    from sklearn.model_selection import StratifiedShuffleSplit
    with open("params.yaml", "r") as file:
        params =yaml.safe_load(file)
    test_size= params["fake_review_detection"]["stratified_split"]["test_size"]
    random_state= params["random_state"]

    df= pd.read_pickle(file_path + "training.pkl")
    X= df["review"].values
    y= df["label"].values
    sss= StratifiedShuffleSplit(n_splits=1, test_size= test_size, random_state= random_state)
    sss.get_n_splits(X,y)
    for train_index, test_index in sss.split(X,y):
        X_train, X_test= X[train_index], X[test_index]
        y_train, y_test= y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test

def handpicked_features_creator(df):
    from sklearn.preprocessing import OneHotEncoder
    # Onehot encoding the category column to return an encoed_cat column
    enc= OneHotEncoder(handle_unknown= "error", sparse= False)
    cat= df["category"].to_numpy().reshape(-1,1)
    enc.fit(cat)
    enc_cat= enc.transform(cat) 
    df["encoded_cat"]= [np.array([vec for vec in enc_cat[i,:] ]) for i in range(len(enc_cat))]
    #getting the length of every review
    df["len_review"]= df["review"].map(len)
    df["group1"]= [np.array([float(rating), float(verified), float(len_review)])\
        for rating, verified, len_review in zip(df["rating"], df["verified"], df["len_review"])]
    df["handpicked_features"]= [np.hstack((df["encoded_cat"][i],df["group1"][i])) for i in range(len(df))]
    return df

def wordvec_features_creator(df):
    from word2vec import get_pretrained_model, generate_dense_features
    """wordvec_features is a helper function that initiated a pretrained wordvec model, runs it on the consolidated_text_column and
    returns a features column of the either the mean or the full wordvec array for the every product

    Args:
        df (data frame): target data frame with a consolidated_text_column 

    Returns:
        data frame: data frame with 
    """
    with open("params.yaml", "r") as file:
        params= yaml.safe_load(file)
    word2vec_model_name= params["load_prepare_fake"]["word2vec_model_name"]
    
    #Get and initialize pretrained word2vec model
    word2vec_model= get_pretrained_model(word2vec_model_name)
    #creating wordvec columns to the df
    df["word2vec_features"]= df["vectorized_reviews"].apply(lambda text: generate_dense_features(tokenized_text= text, model= word2vec_model, use_mean= True))
    return df  

def fake_tfidf_vectorizer_arr(X_train, X_val, X_test, min_df, max_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
        
    train_corpus= [" ".join(lst) for lst in X_train]
    val_corpus= [" ".join(lst) for lst in X_val]
    test_corpus= [" ".join(lst) for lst in X_test]
    vectorizer= TfidfVectorizer(max_df=max_df, min_df=min_df)
    #create vecs from trained vectorizer on train coprus
    vectorizer.fit(train_corpus)
    train_vec= vectorizer.transform(train_corpus)
    val_vec= vectorizer.transform(val_corpus)
    test_vec= vectorizer.transform(test_corpus)
    train_arr= [np.array(i) for i in zip(*train_vec.toarray().T)]
    val_arr= [np.array(i) for i in zip(*val_vec.toarray().T)]
    test_arr= [np.array(i) for i in zip(*test_vec.toarray().T)]
    return train_arr, val_arr, test_arr , vectorizer

def fake_run_pca_arr(X_train,X_val,X_test, n_components):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
        
    #initializing StandardScaler
    scaler= StandardScaler()
    #scaling the array
    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)
    #initializing pca and generating components for X_train
    pca_= PCA(n_components= int(n_components))
    pca_.fit(X_train_arr)
    pc_train= pca_.transform(X_train_arr)
    explained_variance_ratio= pca_.explained_variance_ratio_
    #apply dimensionality reduction to X_test & X_val
    pc_val = pca_.transform(X_val_arr)
    pc_test = pca_.transform(X_test_arr)
    print("total explained variance from {} PCAs = ".format(n_components), np.sum(explained_variance_ratio))
    
    return pc_train, pc_val, pc_test, pca_