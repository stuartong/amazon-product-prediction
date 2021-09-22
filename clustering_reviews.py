import numpy as np
import pandas as pd

import yaml
from pathlib import Path
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import matplotlib.pyplot as plt
import seaborn as sns

import os
import warnings
warnings.filterwarnings('ignore')

from preprocess_data_module import clean_text


def run_clustering(df_path):
    '''
    Function to run clustering models on data
    
    Inputs:
    
    REQUIRED
    1. df - cleaned df with metrics and classes in place (dataframe)
    2. model_type - model type (sklearn class without'()')
    3. params - hyperparameters to use in model (dict)
    OPTIONAL
    4. scale - minmax scale all features (True/False, default: False)
    
    
    Tested on model_types:
    1. KMeans
    2. MiniBatchKMeans
    3. DBSCAN
    
    Depending on the model used, don't forget to import the models BEFORE calling this script
    and adding to imports in the function
    Input: Cleaned df with metrics and classes in place
    Output: Model, clustering result, topics
    '''
    
    model_dict= dict(
        KMeans = KMeans,
        MiniBatchKMeans = MiniBatchKMeans,
        DBSCAN = DBSCAN,
        )
    
    import yaml
    with open("params.yaml", "r") as file:
        param_file= yaml.safe_load(file)
        
    model_type= model_dict[param_file['clustering_model']['model_type']]
    params = param_file['clustering_model']['params']
    scale = param_file['clustering_model_model']['scale']
    
    def clustering_reviews():
    #LOAD DATA
    # (Confirm with Moutaz)
    # def load_df(df):
    #     load_data.py

    #IMPORT DATA
    df= pd.read_pickle('data/reviews.pkl')
    
    # #IMPORT DATA
    # df= pd.read_pickle(df_path+ "/reviews.pkl") 
    
    features = ['reviewText', 'summary']
    for col in [features]:
        df[col]= df[col].replace(np.nan, " ")
    #consolidating review and summary review text  
    df['features']= (df['reviewText'].str.split() + df['summary'].str.split()).map(pd.unique)
    print("cleaning...")
    df['vectorized_reviews']= df['features'].map(clean_text)
    print("cleaning complete!")
    
    #TF-IDF STAGE
    # (Confirm with Moutaz, tweaked from fd_test_model.py)
    # (Sheila does not need options for word2vec, as Sheila will only uses tf-idf in clustering)
    print("starting tfidf transformation...")
    corpus= [" ".join(lst) for lst in df["vectorized_reviews"]]
    tfidf_vectorizer = TfidfVectorizer()  # (Sheila needs min_df, max_df, and possibly max_features, stopwords, and ngram_range)
    tfidf= tfidf_vectorizer.fit_transform(corpus)             
    df["tfidf"]= [np.array(i) for i in zip(*tfidf_vec.toarray().T)]        
    print("tfidf transformation done!")
    
    #SCALE STAGE
    # if (params.get('clustering_model').get('scale')) & (params.get('clustering_model').get('pca') == False) :
    #     scaler_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("scaler_fitted_path")), "rb"))
    #     test_array= scaler_fitted_model.transform(test_arr)
        
        
    def clustering(X, model, reducer, scale=True):
    
    #SCALE
    if scale==True:
        ss = StandardScaler()
        X = ss.fit_transform(X)
    
    #REDUCER
    #reducer (n_components, random_state=42)
    if reducer=='PCA':
        n_components=3
        reducer = PCA(n_components=n, random_state=42)
        reduced_features = reducer.fit_transform(X)
    if reducer=='LSA':
        n_components=3
        reducer=TruncatedSVD(n_components, random_state=42)
        reduced_features = reducer.fit_transform(X)
#     else:
#         n_components=3
#         reducer = NMF(n_components, random_state=42)
#         reduced_features = reducer.fit_transform(X)
    
    #MODEL
    #only DBSCAN has different params
    if model=='KMeans':
        k=6
        model = KMeans(n_clusters=k, random_state=42).fit(X)

    if model=='MiniBatchKmeans':
        k=6
        model = MiniBatchKmeans(n_clusters=k, random_state=42).fit(X)

    else:
        eps=0.02
        min_samples=3  #rule of thumb for min_samples: 2*len(cluster_df.columns)
        model = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(X)
        y_pred = db.fit_predict(X)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Silhouette Coefficient: %0.3f"
            % silhouette_score(X, labels))
        #     plt.scatter(reduced_features[:,0], reduced_features[:,1],c=y_pred, cmap='Paired')
        #     plt.title('DBSCAN')
        #     plt.plot()

        df = pd.DataFrame(reduced_features, columns=['PC1','PC2','PC3'])
        df['labels'] = y_pred
        nn_df = df[df['labels']!=-1]
        plt.scatter(nn_df['PC1'], nn_df['PC2'], c=nn_df['labels'], cmap='Paired')
        plt.title('DBSCAN - No Noise')
        plt.plot()
        
        
    #EVALUATION ALONE
    from numpy import unique
    from numpy import where
    # fit model and predict clusters
    cluster = model.fit_predict(cluster_df)
    # retrieve unique clusters
    clusters = unique(clusters)
    # Calculate cluster validation metrics
    score_silhouette = silhouette_score(cluster, clusters, metric='cosine')
    score_calinski = calinski_harabasz_score(cluster, clusters)
    score_david = davies_bouldin_score(cluster, clusters)
    print('Silhouette Score: %.4f' % score_silhouette)
    print('Calinski Harabasz Score: %.4f' % score_calinski)
    print('Davies Bouldin Score: %.4f' % score_david)