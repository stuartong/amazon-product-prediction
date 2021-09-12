from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def run_pca(array):

    import yaml
    with  open("params.yaml", "r") as file:
        params= yaml.safe_load(file)
        n_components= params["pca"]["n_components"]
    #initializing StandardScaler
    scaler= StandardScaler()
    #scaling the array
    target_arr= StandardScaler().fit_transform(array)
    #initializing pca and generating components
    pca= PCA(n_components= n_components)
    principal_components= pca.fit_transform(target_arr)
    #returning an array of concatenated components 
    return np.array([component for component in principal_components])

