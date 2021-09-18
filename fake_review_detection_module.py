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


    