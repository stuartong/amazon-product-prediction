import gensim
import numpy as np
import tqdm
import pandas as pd

def generate_dense_features(tokenized_texts, model_vectors= None, use_mean= True):
    if model_vectors == None:
        print("""Must chose a pretrained model
              Try:
              - gensim.models.KeyedVectors.load('assets/wikipedia.100.word-vecs.kv',
              - gensim.models.KeyedVectors.load('glove-wiki-gigaword-300'),
              - gensim.models.KeyedVectors.load('word2vec-google-news-300'),
              or type print(list(gensim.downloader.info()['models'].keys())) to get all available list in gensim
              """)
    else:
        target_list= []
        for item in tqdm(tokenized_texts):
            words= [word for word in item if word in model_vectors.vocab]
        if len(words) > 0:
            if use_mean == True:
                #get the mean of the word2vec of every word in tokenized_text
                feature= np.mean(model_vectors[words], axis= 0)
                target_list.append(feature)
            else:
                #just append the full word2vec
                target_list.append([model_vectors[word] for word in words])
        else:
            #just append zeros with the same vector dimension
            target_list.append(model_vectors.vector_size)
    return np.array(target_list)
            