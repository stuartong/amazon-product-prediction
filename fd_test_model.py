import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import pickle
from fake_review_detection_module import wordvec_features_creator
from preprocess_data_module import clean_text
import re
from pathlib import Path


def detect_fake_reviews():
    with open("params.yaml", "r") as file:
        params= yaml.safe_load(file)



    #Importing models/vectorizers ..etc
    if params.get("fd_supervised_model").get("tfidf"):
        tfidf_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("tfidf_path")), "rb"))
    if params.get("fd_supervised_model").get("pca"):
        pca_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("pca_fitted_path")), "rb"))
    if params.get("fd_supervised_model").get("scale"):
        scaler_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("scaler_fitted_path")), "rb"))
    model_fitted= pickle.load(open(Path(params.get("fd_test_stage").get("model_fitted")), "rb"))

    #creating a dictionary of steps to run on the data
    steps_dict= dict(
        tfidf= "tfidf",
        word2vec_features= "word2vec_features",
        )

    #IMPORT DATA
    df= pd.read_pickle("data/prepared/reviews.pkl")
    for col in ["reviewText", "summary"]:
        df[col]= df[col].replace(np.nan, " ")
    #consolidating review and summary review text  
    df["consolidated_text"]= (df["reviewText"].str.split() + df["summary"].str.split()).map(pd.unique)
    print("cleaning...")
    df["vectorized_reviews"]= df["consolidated_text"].map(clean_text )
    print("cleaning complete!")
    
    #WORD2VEC STAGE
    if params.get("load_prepare_fake").get("features").get("word2vec_features"):
        #replacing NaNs with empty lists
        df["vectorized_reviews"]= df["vectorized_reviews"].fillna(" ")
        df= wordvec_features_creator(df)
    #dummy df to fix empty wordvecs with np.zeros since the dtypes are a challenge
    df2= pd.DataFrame({"col": [np.zeros((300,)) for i in range(len(df))]})
    for ind in df[df["vectorized_reviews"].map(len)==0].index:
        df.loc[ind, "word2vec_features"]= str(df2.loc[ind, "col"])
    df["word2vec_features"]=df["word2vec_features"].apply(lambda row: np.array(re.findall(r'\d+', row), dtype= float) if type(row)== str else row)
    
    #TFIDF STAGE
    if params.get("fd_supervised_model").get("tfidf"):
        print("starting tfidf transformation...")
        tfidf_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("tfidf_path")), "rb"))
        corpus= [" ".join(lst) for lst in df["vectorized_reviews"]]
        tfidf_vec= tfidf_fitted_model.transform(corpus)
        df["tfidf"]= [np.array(i) for i in zip(*tfidf_vec.toarray().T)]        
        print("tfidf transformation done!")

    #HANDPICKED STAGE
    if params.get("load_prepare_fake").get("features").get("handpicked_features"):
        df["verified"]= np.where(df["verified"]==True, 1, 0)
        df["len_review"]= df["reviewText"].map(len)
        df["handpicked_features"]= [np.array([float(rating), float(verified), float(len_review)]) \
            for rating, verified, len_review in zip(df["overall"], df["verified"], df["len_review"])]

    #CONSOLIDATE ALL CHOSEN FEATURES
    #creating a dictionary of steps to run on the data
    val_dict=dict(
            tfidf= params.get("fd_supervised_model").get("tfidf"),
            word2vec_features= params.get("load_prepare_fake").get("features").get("word2vec_features"),
            handpicked_features= params.get("load_prepare_fake").get("features").get("handpicked_features")
            )
    print("creating a concatenated features column...")
    df["features"]= df.loc[:, [keys for keys, vals in val_dict.items() if vals]].apply(lambda row: [i for lst in np.array(row) for i in lst], axis=1 )
    print("features column created!")

    #EXTRACTING TEST ARRAY
    test_arr= np.stack(df["features"], axis=0)
    print("Extracting test array of shape {}".format(test_arr.shape))

    #PCA STAGE
    if params.get("fd_supervised_model").get("pca"):
        pca_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("pca_fitted_path")), "rb"))
        scaler_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("scaler_fitted_path")), "rb"))
        scaler_fitted_model.transform(test_arr)
        test_array= pca_fitted_model.transform(test_arr)

    #SCALE STAGE
    if (params.get("fd_supervised_model").get("scale")) & (params.get("fd_supervised_model").get("pca") == False) :
        scaler_fitted_model= pickle.load(open(Path(params.get("fd_test_stage").get("scaler_fitted_path")), "rb"))
        test_array= scaler_fitted_model.transform(test_arr)

    #DETECTING & FILTERING FAKE REVIEWS STAGE
    print("Identifying fake reviews..")
    model_fitted= pickle.load(open(Path(params.get("fd_test_stage").get("model_fitted")), "rb"))
    predictions= model_fitted.predict(test_array)
    print(f"identified {sum(predictions==0)} fake reviews")
    print("filtering the fake reviews")
    df["preds"]= predictions
    true_df= df[df["preds"] ==1]
    return true_df, predictions

if __name__ == "__main__":

    true_df, pred=  detect_fake_reviews()
    Path("data/fake/fake_free_data").mkdir(parents= True, exist_ok=True)
    print('Pickling fake_free_reviews')
    # pickle.dump(true_df, open("data/fake/fake_free_data/fake_free_reviews.pkl",'wb'))
    np.save('data/fake/fake_free_data/fake_free_reviews.npy', pred)
    print('done pickling fake_free_reviews')
    # true_df.to_pickle("data/fake/fake_free_data/fake_free_reviews.pkl")
    np.save('data/fake/fake_free_data/fake_reviews_predictions.npy',pred)
    # pred.to_csv("data/fake/fake_free_data/fake_reviews_predictions.npy", index= False)