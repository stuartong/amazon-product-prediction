import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

'''
    This function taking in the corpus that is a list of text (sentences),
    or list of strings
    and appends the results to the tfidf features. 
    
    Inputs:
    1. corpus: per category review data, list of strings
    
    Outputs:
    1. tfidf features
'''

def count_vectorizer(corpus):
    corpus = list(text)
    vectorizer = TfidfVectorizer()
    
    tfidf = vectorizer(max_features = 6000) #example, please change this accordingly
    tfidf.fit(corpus)
    tfidf_features = tfidf.transform(corpus)
    
    #tfidf = TfidfVectorizer.fit_transform(corpus)
    
    return tfidf_features
