from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def run_pca(df):
    df= df[df["features"].map(len)> 50]
    target_arr= np.hstack(df["features"].values).reshape(-1,304)
    import yaml
    with  open("params.yaml", "r") as file:
        params= yaml.safe_load(file)
        n_components= params["run_pca"]["n_components"]
    
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