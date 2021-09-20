import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from split_data import split_data
from imblearn.over_sampling import SVMSMOTE
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import pickle
from preprocess_data_module import tfidf_vectorizer_arr
from run_pca import run_pca_arr
from pathlib import Path

def run_model(df_path):
    '''
    Function to run classifier model on data
    
    Inputs:
    
    Required
    1. df - cleaned df with metrics and classes in place (dataframe)
    2. model_type - model type (sklearn class without'()')
    3. params - hyperparameters to use in model (dict)
    Optional
    4. scale - minmax scale all features (True/False, default: True)
    5. oversample - SVMSMOTE to over/undersample data for class imbalance (True/False, default: False)
    6. gscv - Run GridSearchCV - time consuming (True/False, default: False)
    7. param_dict - params to tune for gridsearch (dict, default: None)
    8. n_jobs - speed up gridsearch (default - 1 (no parallel)) - use -1 for all available CPU (int)
    9. return_train_score - True/False to effect of params on score - computationally expensive (True/False, defualt: False)


    Tested on model_types:
    1. LogisticRegression
    2. SVC
    3. GradientBosstingClassifier
    4. MLPClassifier
    
    Depending on the model used, don't forget to import the models BEFORE calling this script
    and adding to imports in the function

    Input: Cleaned df with metrics and classes in place
    Output: Model, split and transformed data for scoring function
    '''
     # import models 


    model_dict= dict(
        LogisticRegression= LogisticRegression,
        SVC= SVC,
        GradientBoostingClassifier= GradientBoostingClassifier,
        MLPClassifier= MLPClassifier,
        AdaBoostClassifier=AdaBoostClassifier,
        DecisionTreeClassifier=DecisionTreeClassifier
        )
    
    import yaml
    with open("params.yaml", "r") as file:
        param_file= yaml.safe_load(file)
        
    model_type= model_dict[param_file["supervised_model"]["model_type"]]
    ada_base_model = param_file["supervised_model"]['ada_base_model']
    ada_base_iter = param_file["supervised_model"]['ada_base_iter']
    params= param_file["supervised_model"]["params"] 
    scale= param_file["supervised_model"]["scale"]
    oversample= param_file["supervised_model"]["oversample"]
    gscv= param_file["supervised_model"]["gscv"]
    param_dict= param_file["supervised_model"]["param_dict"]
    n_jobs= param_file["supervised_model"]["n_jobs"]
    return_train_score= param_file["supervised_model"]["return_train_score"]
    split= param_file["supervised_model"]["split"]
    ada_max_depth = param_file["supervised_model"]["ada_max_depth"]
    tfidf= param_file["supervised_model"]["tfidf"]
    min_df= param_file["tfidf_product_success"]["min_df"]
    max_df= param_file["tfidf_product_success"]["max_df"]
    pca= param_file["supervised_model"]["pca"]
    pca_n_components= param_file["supervised_model"]["pca_n_components"]
    if model_type == AdaBoostClassifier:
        param_dict["base_estimator"]= [model_dict[ada_base_model](max_depth= i) for i in ada_base_iter]
        params["base_estimator"]= model_dict[ada_base_model](max_depth=ada_max_depth)

   
    #import data
    df= pd.read_pickle(df_path+ "/products.pkl") 
    
    feature = 'features'

    # list of models where feature scaling is recommended
    scaling_recommended = [LogisticRegression,SVC]

    # check and drop feature matrix that aren't full
    median_vec = df[feature].map(len).median()
    empty = len(df[df[feature].map(len)!=median_vec])
    print(f"Dropping {empty} rows with empty/semi-full vectors")
    #adding the tfidf
    if tfidf:
        
        df = df.loc[df[feature].map(len)==median_vec].copy()
        # aligning vector arrays for sklearn
        # df[feature]=df[feature].apply(lambda x: x[0])

        # split the data to train/test/validate split
        # and get X and y for splits
        # change string 'wordvec' to string 'features' once moutaz fixes code
        train, val, test = split_data(df,split)

        X_train = train[[feature, "consolidated_text_column"]]
        y_train = list(train['class_label'])

        X_test = test[[feature, "consolidated_text_column"]]
        y_test = list(test['class_label'])

        X_val = val[[feature, "consolidated_text_column"]]
        y_val = list(val['class_label'])
        
        print("Creating Tfidf vectors for train and test data...")
        X_train.loc[:,"consolidated_text_column"],X_val.loc[:,"consolidated_text_column"], X_test.loc[:,"consolidated_text_column"]= tfidf_vectorizer_arr(X_train= X_train.loc[:,"consolidated_text_column"].values,
                                                                                                                                                            X_val= X_val.loc[:,"consolidated_text_column"].values,
                                                                                                                                                            X_test= X_test.loc[:,"consolidated_text_column"].values,\
                                                                                                                                                            min_df=min_df, max_df= max_df)
        print("tfidf vectors created!")
        X_train= np.array([(np.array([vec for lst in X_train.values[i].flatten("C") for vec in lst])) for i in range(len(X_train))])
        X_val  = np.array([(np.array([vec for lst in X_val.values[i].flatten("C") for vec in lst])) for i in range(len(X_val))])
        X_test = np.array([(np.array([vec for lst in X_test.values[i].flatten("C") for vec in lst])) for i in range(len(X_test))])
        
    else:        
        df = df[df[feature].map(len)==median_vec].copy()

        # aligning vector arrays for sklearn
        # df[feature]=df[feature].apply(lambda x: x[0])

        # split the data to train/test/validate split
        # and get X and y for splits
        # change string 'wordvec' to string 'features' once moutaz fixes code
        train, val, test = split_data(df,split)

        X_train = np.stack(train[feature],axis=0)
        y_train = list(train['class_label'])

        X_test = np.stack(test[feature],axis=0)
        y_test = list(test['class_label'])

        X_val = np.stack(val[feature],axis=0)
        y_val = list(val['class_label'])

    # if scale is True, we will fit and transform all X
    # with a MinMaxScaler
    if (scale==True) | (model_type in scaling_recommended):
        print('Scaling features...')
        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)
    else:
        print('No scaling done')

    # if oversample is True, we will use SMOTEEN to oversample
    # minority class only on the training data
    # oversampling generally for less data - considering undersampling but risks overfitting
   
    if oversample==True:
        print('Oversampling Data')
        print('Original dataset shape:', Counter(y_train))
        
        smote = SVMSMOTE(random_state=0)
        X_train, y_train = smote.fit_resample(X_train,y_train)
        print('Resample dataset shape:', Counter(y_train))
    else:
        print('No oversampling done')
        print('Current Class Balance:', Counter(y_train))

    if pca:
        print("Generating PCAs to reduce data dimensions...")
        X_train, X_val, X_test = run_pca_arr(X_train,X_val,X_test,n_components= pca_n_components)
        print("PCA process done!")
        
    
    # gridsearch/fit classifier
    
    if gscv == False:
        clf = model_type(**params).fit(X_train,y_train)
    elif (gscv==True) & (param_dict==None):
        print('GridSearchCV requires param_dict in order to run')
        clf = model_type(**params).fit(X_train,y_train)
    elif (gscv==True) & (len(param_dict)>=1):
        print('Running GridSearch and fitting model with the best params')
        model=model_type()
        CV_clf = GridSearchCV(model,
                              param_dict,
                              scoring='accuracy',
                              return_train_score=return_train_score,
                              n_jobs=n_jobs,
                              verbose=3
                              )
        CV_clf.fit(X_train,y_train)
        clf = CV_clf.best_estimator_
        best_score = CV_clf.best_score_
        best_params = CV_clf.best_params_
        print(f'Best Score for CV: {best_score}')
        print(f'Best Parameters: {best_params}')
        if return_train_score==True:
            print(CV_clf.cv_results_)
        else:
            pass

    return clf,X_train,X_test,X_val,y_train,y_test,y_val

if __name__ == "__main__":
    import argparse
    parser= argparse.ArgumentParser()
    parser.add_argument("df_path", help= "target dataframe path (str)")
    args= parser.parse_args()
    
    clf,X_train,X_test,X_val,y_train,y_test,y_val = run_model(args.df_path)
    
    Path("model/product_success").mkdir(parents=True, exist_ok=True)
    pickle.dump(clf, open("model/product_success/model.pkl", "wb"))
    np.save("model/product_success/X_train.npy", X_train)
    np.save("model/product_success/X_test.npy", X_test)
    np.save("model/product_success/X_val.npy", X_val)
    np.save("model/product_success/y_train.npy", y_train)
    np.save("model/product_success/y_test.npy", y_test)
    np.save("model/product_success/y_val.npy", y_val)