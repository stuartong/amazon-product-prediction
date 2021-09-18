from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def run_pca_df(df):
    import yaml
    with  open("params.yaml", "r") as file:
        params= yaml.safe_load(file)
        n_components= params["run_pca"]["n_components"]
    df= df[df["features"].map(len)> n_components]
    array_len= max(df["features"].map(len))
    target_arr= np.hstack(df["features"].values).reshape(-1,array_len)
    #initializing StandardScaler
    scaler= StandardScaler()
    #scaling the array
    target_arr= scaler.fit_transform(target_arr)
    #initializing pca and generating components
    pca_= PCA(n_components= int(n_components))
    principal_components= pca_.fit_transform(target_arr)
    df1= pd.DataFrame(principal_components).apply(list, axis=1)
    df.reset_index(inplace= True)
    df["features"]= df1
    explained_variance_ratio= pca_.explained_variance_ratio_
    # print("pca explained variance ration= ", explained_variance_ratio)
    print("total explained variance from {} PCAs = ".format(n_components), np.sum(explained_variance_ratio))
    #returning an array of concatenated components 
    return df

def run_pca_arr(X_train,X_val,X_test, n_components):
        
    #initializing StandardScaler
    scaler= StandardScaler()
    #scaling the array
    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)
    #initializing pca and generating components for X_train
    pca_= PCA(n_components= int(n_components))
    pc_train= pca_.fit_transform(X_train_arr)
    explained_variance_ratio= pca_.explained_variance_ratio_
    #apply dimensionality reduction to X_test & X_val
    pc_val = pca_.transform(X_val_arr)
    pc_test = pca_.transform(X_test_arr)
    print("total explained variance from {} PCAs = ".format(n_components), np.sum(explained_variance_ratio))
    #returning an array of concatenated components 
    return pc_train, pc_val, pc_test