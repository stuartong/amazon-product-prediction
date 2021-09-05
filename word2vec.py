import gensim
import numpy as np
import tqdm
import pandas as pd

def get_pretrained_model(target_model= 'glove-wiki-gigaword-300' ):
    """get_pretrained_model is a helper function to download the pretrained model

    Args:
        target_model (str, optional): Target pretrained model. Defaults to 'glove-wiki-gigaword-300'.

    Returns:
        pretrained KeyedVector
    """
    import gensim.downloader as api
    return api.load(target_model)

def generate_dense_features(tokenized_texts, model= None, use_mean= True):
    """This function takes tokenized_texts in list format and a pretrained word2vec model
    and returns an array of either a vector of word embeddings or the mean of that vector
    depending on the users choice
    
    Args:
        tokenized_texts (list): list of strings to match agains the pretrained word2vec model
        model (matrix): matrix of words and their respective word embedding vectors
        use_mean (bool, optional): True to get the mean of the vector, False to return full vector. Default to 'True'

    Returns:
        vector: vector/ or average of the vector of embedding weights for every word"""
    if model == None:
        print("""Must chose a pretrained model
              Try:
              - gensim.models.KeyedVectors.load('assets/wikipedia.100.word-vecs.kv',
              - gensim.models.KeyedVectors.load('glove-wiki-gigaword-300'),
              - gensim.models.KeyedVectors.load('word2vec-google-news-300'),
              or type print(list(gensim.downloader.info()['models'].keys())) to get all available list in gensim
              """)
    else:
        
        target_list= []
        for item in tokenized_texts:
            words= [word for word in item if model.index_to_key]
        if len(words) > 0:
            if use_mean == True:
                #get the mean of the word2vec of every word in tokenized_text
                feature= np.mean(model[words], axis= 0)
                target_list.append(feature)
            else:
                #just append the full word2vec
                target_list.append([model[word] for word in words])
        else:
            #just append zeros with the same vector dimension
            target_list.append(model.vector_size)
    return np.array(target_list)
            