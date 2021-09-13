from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def run_pca(array):
    target_arr= array.reshape(-1,1)
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
    #returning an array of concatenated components 
    return np.array([component for component in principal_components])

