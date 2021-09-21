import pandas as pd 
import numpy as np
import yaml
import time

def main():
    df =pd.read_pickle("data/prepared/reviews.pkl")
    from preprocess_data_module import clean_text
    from fake_review_detection_module import wordvec_features_creator
    df["consolidated_text"]= df["reviewText"].str.split() + df["summary"].str.split()
    print("cleaning...")
    import concurrent.futures
    start= time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        lst= executor.map(clean_text, df["consolidated_text"])
    # df["vectorized_reviews"]=  pd.Series(lst)
    print("cleaning complete!")
    end= time.perf_counter()
    print(f'finished in {round(end-start,2)} seconds ')
    from word2vec import get_pretrained_model, generate_dense_features
    with open("params.yaml", "r") as file:
            params= yaml.safe_load(file)
    word2vec_model_name= params["load_prepare_fake"]["word2vec_model_name"]
    word2vec_model= get_pretrained_model(word2vec_model_name)
    df["vectorized_reviews"]= df["vectorized_reviews"].replace(np.nan, " ")
    def do_something(item):
        return [word for word in item if word in word2vec_model.index_to_key]
    # import concurrent.futures
    # start= time.perf_counter()
    # print("multiprocessing started...")
    # lst= df["vectorized_reviews"][:1000]
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     word_lst= executor.map(do_something, lst)
    # end= time.perf_counter()
    # print(f'finished in {round(end-start,2)} seconds ')
    # return word_lst

if __name__ == "__main__":
    df= main()
    with open("test.txt", "w") as file:
        [file.write(word ) for line in df for word in line]