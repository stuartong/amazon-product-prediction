import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

def run_kmean_arr(train_df,val_df,test_df, X_train,X_test,X_val,best_k='search'):
    '''
    Function to fit a dbscan object on training data and cluster labels
    Use fitted dbscan object to predict X_val and X_test cluster labels

    Input:
    Full dataframe splits: train_df,test_df,val_df
    Existing Feature Array: X_train,X_test,X_val

    Returns X_train,X_val,X_test array for use downstream
    '''

    # get arrays to fit
    train_arr = np.stack(train_df.word2vec_features,axis=0)
    test_arr = np.stack(test_df.word2vec_features,axis=0)
    val_arr = np.stack(val_df.word2vec_features,axis=0)

    # scale arrays
    scaler = StandardScaler().fit(train_arr)
    train_arr = scaler.transform(train_arr)
    test_arr = scaler.transform(test_arr)
    val_arr = scaler.transform(val_arr)

    # initially planned to use a dbscan but have to do a workaround 
    # to predict train and validation clusters - not ideal
    # https://stackoverflow.com/questions/27822752/scikit-learn-predicting-new-points-with-dbscan

    # define cluster range
    # as we are trying to cluster sub-categories - expect higher number of clusters
    range_n_clusters = [200,400,600,800,1000]

    if best_k=='search':
        print("Checking for best k...this could take several minutes... ")
        # get best k using silhoutte score
        best_score = 0
        best_k = 0
        for idx,n_clusters in enumerate(range_n_clusters):
            print(f"Iteration {idx+1} of {len(range_n_clusters)} for cluster size {n_clusters}...")
            # create k-mean cluster object using n_clusters
            k_cluster = KMeans(n_clusters=n_clusters,random_state=42)
            cluster_labels = k_cluster.fit_predict(train_arr)

            # get scores to evaluate clusters
            silhouette_avg = silhouette_score(train_arr, cluster_labels)
            inertia_score = k_cluster.inertia_
            print(f'For {n_clusters}, the average silhouette score is {silhouette_avg} and inertia is {inertia_score}')
            
            # check best score and update best k
            if silhouette_avg > best_score:
                best_k = n_clusters
            else:
                pass
        print(f"Done finding the best k. The optimal k is ~{best_k}.")
    else:
        best_k=best_k
    
    print("Fitting Best K... ")
    # refit the cluster object using best_k
    k_cluster = KMeans(n_clusters=best_k,random_state=42)
    k_cluster.fit(train_arr)

    # get cluster labels on fitted cluster
    train_cl = k_cluster.predict(train_arr)
    test_cl = k_cluster.predict(test_arr)
    val_cl = k_cluster.predict(val_arr)

    # append labels to existing pre-processed feature array
    X_train = np.c_[X_train,train_cl]
    X_test = np.c_[X_test,test_cl]
    X_val = np.c_[X_val,val_cl]

    return X_train,X_test,X_val










