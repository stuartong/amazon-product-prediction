import gensim
import numpy as np
import tqdm
import pandas as pd

def get_pretrained_model(target_model= 'glove-wiki-gigaword-300' ):
    """get_pretrained_model is a helper function that checks if the target_model is already available in the gensim download folder in
    the base directory, which is automatically created when downloaded through the gensim downloader api. If so, it initializes the model
    otherwise it downloads the pretrained model then initializes it.

    Args:
        target_model (str, optional): Target pretrained model. Defaults to 'glove-wiki-gigaword-300'.

    Returns:
        pretrained KeyedVector
    """
    from pathlib import Path
    import gensim.downloader as api
    from gensim.downloader import base_dir
    import os
    from gensim.models import KeyedVectors
    #getting the target path where the model is usually downloaded in by gensim
    path = os.path.join(base_dir, target_model, target_model +".gz")
    print("checking if pretrained model already downloaded")
    if Path(path).exists():
        print("{} model is already downloaded".format(target_model))
        print("Initializing model")
        model = KeyedVectors.load_word2vec_format(path)
    else:
        print("{} model is not found in gensim default base directory".format(target_model))
        print("Downloading {} model. \n Make yourself a nice cup of tea, this may take a while!".format(target_model))
        download_file= api.load(target_model)
        print("Download Complete!")
        print("Initializing model")
        model= KeyedVectors.load_word2vec_format(path)    

    ### TRY THE DVC OPEN FILE USING PYTHON API instead of downloding it
    #https://dvc.org/doc/start/data-and-model-access
    return model

def generate_dense_features(tokenized_text, model= None, use_mean= True):
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
        for item in tokenized_text:
            words= [word for word in item if model.index_to_key]
        if len(words) > 0:
            if use_mean == True:
                try:
                    #get the mean of the word2vec of every word in tokenized_text
                    feature= np.mean([model[word] for word in words] , axis= 0)
                    target_list.append(feature)
                except:
                    pass
            else:
                try:
                    #just append the full word2vec
                    target_list.append([model[word] for word in words])
                except:
                    pass
        else:
            #just append zeros with the same vector dimension
            target_list.append(model.vector_size)
    return np.array(target_list)
            