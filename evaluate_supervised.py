import pandas as pd
import numpy as np
from sklearn.metrics import plot_precision_recall_curve, f1_score, matthews_corrcoef,confusion_matrix, classification_report,plot_confusion_matrix,recall_score,plot_roc_curve, roc_curve, roc_auc_score, accuracy_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import json 
import os
import sys
from sklearn import metrics
import pickle

def evaluate_model():
    #import params
    import yaml
    with open("params.yaml", "r") as file:
        params= yaml.safe_load(file)
    f1_avg= params["evaluate_supervised"]["f1_avg"]
    #loading model, train, eval, test files
    with open(r"model\product_success\model.pkl", "rb") as model:
        clf= pickle.load(model)
    with open(r"model\product_success\X_train.npy", "rb") as  file:
        X_train= np.load(file)
    with open(r"model\product_success\X_test.npy", "rb") as  file:
        X_test= np.load(file)
    with open(r"model\product_success\X_val.npy", "rb") as  file:
        X_val= np.load(file)
    with open(r"model\product_success\y_train.npy", "rb") as  file:
        y_train= np.load(file)
    with open(r"model\product_success\y_test.npy", "rb") as  file:
        y_test= np.load(file)
    with open(r"model\product_success\y_val.npy", "rb") as  file:
        y_val= np.load(file)

    #importing the model_metrics to include in the report
    model_type= [params["supervised_model"]["model_type"]]
    ada_base_model = params["supervised_model"]['ada_base_model']
    ada_base_iter = params["supervised_model"]['ada_base_iter']
    params_used= params["supervised_model"]["params"] 
    scale= params["supervised_model"]["scale"]
    oversample= params["supervised_model"]["oversample"]
    gscv= params["supervised_model"]["gscv"]
    param_dict= params["supervised_model"]["param_dict"]
    n_jobs= params["supervised_model"]["n_jobs"]
    return_train_score= params["supervised_model"]["return_train_score"]
    split= params["supervised_model"]["split"]
    ada_max_depth = params["supervised_model"]["ada_max_depth"]
    model_params_dict= dict(
        model_type= model_type)
    ada_model_dict= dict(
        ada_base_model= ada_base_model,
        ada_base_iter= ada_base_iter,
        ada_max_depth= ada_max_depth,
        ada_params_dict= param_dict)
    model_params= dict(
        scale= scale,
        oversample= oversample,
        gscv= gscv,
        n_jobs= n_jobs,
        return_train_score= return_train_score,
        split= split,
        )
    model_params.update(params_used)
    #creating an if\else statement to update subparamters only if their parent parameter is True
    if model_type == "AdaBoostClassifier":
        model_params_dict["ada_model_params"]= ada_model_dict
        model_params_dict["params"]= model_params
    else:
        model_params_dict["params"]= model_params
        
    #import features to include in the report
    wordvec= params["preprocess_products"]["word2vec_features"]
    handpicked= params["preprocess_products"]["handpicked_features"]
    wordvec_model= params["preprocess_products"]["word2vec_model_name"]
    pca= params["supervised_model"]["pca"]
    pca_comp= params["supervised_model"]["pca_n_components"]
    tfidf= params["supervised_model"]["tfidf"]
    category_thresh= params["preprocess_products"]["occurrence_threshold"]
    #creating an if\else statement to update subparamters only if their parent parameter is True
    features_dict= dict(
        category_thresh= category_thresh,
        tfidf= tfidf,
        handpicked= handpicked,)
    if wordvec:
        wordvec_dict= dict(wordvec =wordvec,
                        wordvec_model= wordvec_model)
        features_dict.update(wordvec_dict)
    else:
        features_dict.update(wordvec)
    if pca:
        pca_dict=dict(pca= pca, pca_comp= pca_comp)
        features_dict.update(pca_dict)
    else:
        features_dict["pca"]= pca
    
    # get predictions for test set
    y_pred = clf.predict(X_test)
    y_val_pred = clf.predict(X_val)

    # create & fit dummy classifiers for comparison
    dummy_uniform = DummyClassifier(strategy='uniform',random_state=42).fit(X_train,y_train)
    dummy_mf = DummyClassifier(strategy='most_frequent',random_state=42).fit(X_train,y_train)

    # get dummy predictions
    rand_dev_preds = dummy_uniform.predict(X_test)
    mf_dev_preds = dummy_mf.predict(X_test)

    # Use accuracy to check for overfitting and comp with dummy
    print('calculating metrics..')
    train_acc = clf.score(X_train,y_train)
    test_acc = clf.score(X_test,y_test)
    val_acc = clf.score(X_val,y_val)
    dummy_mf_acc = dummy_mf.score(X_test,y_test)
    dummy_uni_acc = dummy_uniform.score(X_test,y_test)
    test_f1 = f1_score(y_test,y_pred,average=f1_avg)
    val_f1 = f1_score(y_val,y_val_pred,average=f1_avg)
    dummy_mf_f1 = f1_score(y_test,mf_dev_preds,average=f1_avg)
    dummy_uni_f1 = f1_score(y_test,rand_dev_preds,average=f1_avg)
    test_mcc = matthews_corrcoef(y_test,y_pred)
    val_mcc = matthews_corrcoef(y_val,y_val_pred)
    dummy_uni_mcc = matthews_corrcoef(y_test,rand_dev_preds)
    test_clf_report= classification_report(y_test,y_pred,output_dict=True)
    val_clf_report= classification_report(y_val,y_val_pred,output_dict=True)

    
    #saving scores to file to be added to dvc metrics
    acc_scores= {"training accuracy": train_acc, 
            "Test accuracy": test_acc,
            "Validation accuracy": val_acc,
            "Dummy most frequent accuracy": dummy_mf_acc,
            "Dummy uniform accuracy": dummy_uni_acc,
            "Test F1-score": test_f1,
            "Validation F1-score": val_f1,
            "Dummy most frequent F1-Score": dummy_mf_f1,
            "Dummy Unified F1-score": dummy_uni_f1,
            "Test Matthews Corrcoef": test_mcc,
            "Validation Matthews Corrcoef": val_mcc,
            "Dummy Unified Mathews Corrcoef": dummy_uni_mcc,
            "Classification Report - Test": test_clf_report,
            "Classification Report - Validation": val_clf_report
                    }
    
    report= dict(
        preprocess_features= features_dict,
        model_specs= model_params_dict,
        scores_report= acc_scores)
    
    print('calculating metrics complete!')
    
     
    with open(sys.argv[1], "w") as fd:
        json.dump(report, fd, indent=4)

      
    #display plots
    # fig , axes = plt.subplots(1,3,figsize=(24,8))
    # axes[0].title.set_text('Confusion Matrix')
    # axes[1].title.set_text('ROC Curve')
    # axes[2].title.set_text('Precision-Recall Curve')

    # cm = plot_confusion_matrix(clf,X_test,y_test,ax=axes[0]);
    # roc_curve = plot_roc_curve(clf,X_test,y_test,ax=axes[1]);
    # pr_curve = plot_precision_recall_curve(clf,X_test,y_test,ax=axes[2]);

    # cm.plot(ax=axes[0]);
    # roc_curve.plot(ax=axes[1]); 
    # pr_curve.plot(ax=axes[2]);


if __name__ == "__main__":
    
    evaluate_model()



