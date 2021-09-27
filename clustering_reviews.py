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
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import matplotlib.pyplot as plt
import seaborn as sns

import os
import warnings
warnings.filterwarnings('ignore')

from preprocess_data_module import clean_text


def run_clustering():
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
    
    import yaml
    with open("params.yaml", "r") as file:
        param_file= yaml.safe_load(file)
        
#     #initializing neptune run
#     run = neptune.init(
#     project = 'Milestone2/MilestoneII',
#     api_token = config('NEPTUNE_API_KEY'),
#     name = 'clustering reviews',
#     tags = ['clustering model', 'cluster reviews']
# )  
        
    data_source= data_source_dict[param_file["clustering_model"]["data_source"]]
    model= model_dict[param_file['clustering_model']['model_type']]
    params = param_file['clustering_model']['params']
    scale = param_file['clustering_model_model']["raw_data_params"]['scale']
    reducer= param_file["clustering_model"]["raw_data_params"]["reducer"]["name"]
    n_components= param_file["clustering_model"]["raw_data_params"]["reducer"]["n_components"]
    tfidf_params= param_file["clustering_model"]["raw_data_params"]["tfidf_params"]

    data_source_dict = dict(
        raw = "data/prepared/reviews.pkl",
        preprocessed = "data/reviews/reviews.pkl"
        )
    model_dict= dict(
        KMeans = KMeans,
        MiniBatchKMeans = MiniBatchKMeans,
        DBSCAN = DBSCAN,
        )
  
    #IMPORT DATA
    if data_source_dict.get(data_source) is not None:
        df= pd.read_pickle(data_source_dict.get(data_source))
    else:
        preds= np.load("data/fake/fake_free_data/fake_free_reviews.npy")
        df= pd.read_pickle("data/prepared/reviews.pkl")
        df["preds"]= preds
        df= df[df["preds"] > 0]
    

    if data_source_dict.get(data_source).key() == "raw":
        features = ['reviewText', 'summary']
        for col in [features]:
            df[col]= df[col].replace(np.nan, " ")
        #consolidating review and summary review text  
        df['features']= (df['reviewText'].str.split() + df['summary'].str.split()).map(pd.unique)
        print("cleaning...")
        df['vectorized_reviews']= df['features'].map(clean_text)
        print("cleaning complete!")
        
        #TF-IDF STAGE
        print("starting tfidf transformation...")
        corpus= [" ".join(lst) for lst in df["vectorized_reviews"]]
        tfidf_vectorizer = TfidfVectorizer(tfidf_params)  # (Sheila needs min_df, max_df, and possibly max_features, stopwords, and ngram_range)
        tfidf_vec= tfidf_vectorizer.fit_transform(corpus)             
        df["tfidf"]= [np.array(i) for i in zip(*tfidf_vec.toarray().T)]        
        print("tfidf transformation done!")
        X= np.stack(df["tfidf"].to_numpy(), axis=0)
        
        #SCALE
        if scale==True:
            ss = StandardScaler()
            X = ss.fit_transform(X)
        
        #REDUCER
        #reducer (n_components, random_state=42)
        if reducer=='NMF':
            n_components=3
            reducer = NMF(n_components, random_state=42)
            reduced_features = reducer.fit_transform(X)
            
        if reducer=='LSA':
            n_components=3
            reducer=TruncatedSVD(n_components, random_state=42)
            reduced_features = reducer.fit_transform(X)
        else:
            n_components=3
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_features = reducer.fit_transform(X)

    
        #PARAMETERS
        #FINDING BEST K OPTION 1
        # inertia = [0,0]

        # for k in range(2, 10):
        #     km = KMeans(n_clusters=k, random_state=42)
        #     km.fit(X)
        #     inertia.append(km.inertia_)

        # best_k = 
        
        #FINDING BEST K OPTION 2
        #model = KMeans()
        #visualizer = KElbowVisualizer(model, k=(4,12))

        #visualizer.fit(X)      
        #visualizer.show();  
        
        #FINDING THE BEST EPS
        #from sklearn.neighbors import NearestNeighbors
        #neigh = NearestNeighbors(n_neighbors=2)
        #nbrs = neigh.fit(X)
        #distances, indices = nbrs.kneighbors(X)
        # distances = np.sort(distances, axis=0)
        # distances = distances[:,1]
        #plt.plot(distances);
        
        #best_eps = 


    #only DBSCAN has different params
    if model=='KMeans':
        k=best_k
        model = KMeans(n_clusters=k, random_state=42).fit(X)

    if model=='MiniBatchKmeans':
        k=best_k
        model = MiniBatchKmeans(n_clusters=k, random_state=42).fit(X)

    else:
        eps=best_eps
        min_samples=3  #rule of thumb for min_samples: 2*len(cluster_df.columns)
        
        model = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(X)
        y_pred = model.fit_predict(X)
        labels = model.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
       
    
        #COMPUTING SILHOUETTE, HARABASZ, and BOULDIN SCORE
        print("Silhouette Score: %0.3f"
            % silhouette_score(X, labels))
        print("Calinski Harabasz Score: %0.3f"
            % calinski_harabasz_score(X, labels))
        print("Davies Bouldin Score: %0.3f"
            % davies_bouldin_score(X, labels))
        
        #PLOTTING THE GRAPH
        #     plt.scatter(reduced_features[:,0], reduced_features[:,1],c=y_pred, cmap='Paired')
        #     plt.title('DBSCAN')
        #     plt.plot();

        df = pd.DataFrame(reduced_features, columns=['PC1','PC2','PC3'])
        df['labels'] = y_pred
        nn_df = df[df['labels']!=-1]
        plt.scatter(nn_df['PC1'], nn_df['PC2'], c=nn_df['labels'], cmap='Paired')
        plt.title('DBSCAN - No Noise')
        plt.plot();
                
    else:
        preds= np.load("data/fake/fake_free_data/fake_free_reviews.npy")
        df= pd.read_pickle("data/prepared/reviews.pkl")
        df["preds"]= preds
        df= df[df["preds"] > 0]
        
        #SCALE
        if scale==True:
            ss = StandardScaler()
            X = ss.fit_transform(X)
        
        #REDUCER
        #reducer (n_components, random_state=42)
        if reducer=='NMF':
            n_components=3
            reducer = NMF(n_components, random_state=42)
            reduced_features = reducer.fit_transform(X)
            
        if reducer=='LSA':
            n_components=3
            reducer=TruncatedSVD(n_components, random_state=42)
            reduced_features = reducer.fit_transform(X)
        else:
            n_components=3
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_features = reducer.fit_transform(X)

    
        #PARAMETERS
        #FINDING BEST K OPTION 1
        # inertia = [0,0]

        # for k in range(2, 10):
        #     km = KMeans(n_clusters=k, random_state=42)
        #     km.fit(X)
        #     inertia.append(km.inertia_)

        # best_k = 
        
        #FINDING BEST K OPTION 2
        #model = KMeans()
        #visualizer = KElbowVisualizer(model, k=(4,12))

        #visualizer.fit(X)      
        #visualizer.show();  
        
        #FINDING THE BEST EPS
        #from sklearn.neighbors import NearestNeighbors
        #neigh = NearestNeighbors(n_neighbors=2)
        #nbrs = neigh.fit(X)
        #distances, indices = nbrs.kneighbors(X)
        # distances = np.sort(distances, axis=0)
        # distances = distances[:,1]
        #plt.plot(distances);
        
        #best_eps = 


    #only DBSCAN has different params
    if model=='KMeans':
        k=best_k
        model = KMeans(n_clusters=k, random_state=42).fit(X)

    if model=='MiniBatchKmeans':
        k=best_k
        model = MiniBatchKmeans(n_clusters=k, random_state=42).fit(X)

    else:
        eps=best_eps
        min_samples=3  #rule of thumb for min_samples: 2*len(cluster_df.columns)
        
        model = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(X)
        y_pred = model.fit_predict(X)
        labels = model.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
       
    
        #COMPUTING SILHOUETTE, HARABASZ, and BOULDIN SCORE
        print("Silhouette Score: %0.3f"
            % silhouette_score(X, labels))
        print("Calinski Harabasz Score: %0.3f"
            % calinski_harabasz_score(X, labels))
        print("Davies Bouldin Score: %0.3f"
            % davies_bouldin_score(X, labels))
        
        #PLOTTING THE GRAPH
        #     plt.scatter(reduced_features[:,0], reduced_features[:,1],c=y_pred, cmap='Paired')
        #     plt.title('DBSCAN')
        #     plt.plot();

        df = pd.DataFrame(reduced_features, columns=['PC1','PC2','PC3'])
        df['labels'] = y_pred
        nn_df = df[df['labels']!=-1]
        plt.scatter(nn_df['PC1'], nn_df['PC2'], c=nn_df['labels'], cmap='Paired')
        plt.title('DBSCAN - No Noise')
        plt.plot();

        
# TOPICS        
def topic_modeling(model, feature_names, no_top_words=10, topic_names=None):
    for index, topic in enumerate(model.components_):
        if not topic_names or not topic_names[index]:
            print(f'\nTopic {index}')
        else:
            print(f'\nTopic {topic_names[index]}:')
        the_topics = ', '.join([f'{feature_names[i]} ({topic[i]:6.4f})' 
                             for i in topic.argsort()[:-no_top_words-1:-1]])
        print(the_topics)
