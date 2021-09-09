import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from split_data import split_data
from imblearn.over_sampling import SMOTE
from collections import Counter

def run_model(df,model_type,params,scale=True,oversample=False):
    '''
    Function to run classifier model

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
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    feature = 'wordvec'

    # list of models where feature scaling is recommended
    scaling_recommended = [LogisticRegression,SVC]

    # check and drop empty vectors
    empty = len(df[df[feature].map(len)==0])
    print(f"Dropping {empty} rows with empty vectors")
    df = df[df[feature].map(len)!=0].copy()

    # aligning vector arrays for sklearn
    df[feature]=df[feature].apply(lambda x: x[0])

    # split the data to train/test/validate split
    # and get X and y for splits
    # change string 'wordvec' to string 'features' once moutaz fixes code
    train, val, test = split_data(df,0.60)

    X_train = np.stack(train['wordvec'],axis=0)
    y_train = list(train['class_label'])

    X_test = np.stack(test['wordvec'],axis=0)
    y_test = list(test['class_label'])

    X_val = np.stack(val['wordvec'],axis=0)
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
        pass

    # if oversample is True, we will use SMOTE to oversample
    # oversampling generally for less data - considering undersampling but risks overfitting
    # minority class only on the training data
    if oversample==True:
        print('Oversampling Data')
        print('Origianl dataset shape:', Counter(y_train))
        
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train,y_train)
        print('Resampple dataset shape:', Counter(y_train))
    else:
        print('No oversampling done')
        print('Current Class Balance:', Counter(y_train))
    
    # fit classifier
    # TO DO INSERT GridSearchCV routine here to get optimised hyperparameters
    clf = model_type(**params).fit(X_train,y_train)

    return clf,X_train,X_test,X_val,y_train,y_test,y_val




