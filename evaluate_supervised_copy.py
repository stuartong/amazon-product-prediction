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
    with open("model/model.pkl", "rb") as model:
        clf= pickle.load(model)
    with open("model/X_train.npy", "rb") as  file:
        X_train= np.load(file)
    with open("model/X_test.npy", "rb") as  file:
        X_test= np.load(file)
    with open("model/X_val.npy", "rb") as  file:
        X_val= np.load(file)
    with open("model/y_train.npy", "rb") as  file:
        y_train= np.load(file)
    with open("model/y_test.npy", "rb") as  file:
        y_test= np.load(file)
    with open("model/y_val.npy", "rb") as  file:
        y_val= np.load(file)
    
    
    # get predictions for test set
    y_pred = clf.predict(X_test)

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
    dummy_mf_acc = dummy_mf.score(X_test,y_test)
    dummy_uni_acc = dummy_uniform.score(X_test,y_test)
    test_f1 = f1_score(y_test,y_pred,average=f1_avg)
    dummy_mf_f1 = f1_score(y_test,mf_dev_preds,average=f1_avg)
    dummy_uni_f1 = f1_score(y_test,rand_dev_preds,average=f1_avg)
    test_mcc = matthews_corrcoef(y_test,y_pred)
    dummy_uni_mcc = matthews_corrcoef(y_test,rand_dev_preds)
    clf_report= classification_report(y_test,y_pred)
    #saving scores to file to be added to dvc metrics
    acc_scores= {"training accuracy": train_acc, 
                    "Test accuracy": test_acc,
                    "Dummy most frequent accuracy": dummy_mf_acc,
                    "Dummy uniform accuracy": dummy_uni_acc,
                    "Test F1-score": test_f1,
                    "Dummy most frequent F1-Score": dummy_mf_f1,
                    "Dummy Unified F1-score": dummy_uni_f1,
                    "Test Mathews Corrcoef": test_mcc,
                    "Dummy Unified Mathews Corrcoef": dummy_uni_mcc,
                    # "Classification Report": clf_report
                    }
    
    print('calculating metrics complete!')
    
    # avg_prec= metrics.average_precision_score(y_train, y_pred) 
    # roc_auc= metrics.roc_auc_score(y_train, y_pred)
    # acc_scores= dict(
    #     avg_prec= avg_prec,
    #     roc_auc= roc_auc)
    #create scores file 
    # score_file= "scores.json"
    # path=  .path.join("eval_metrics", score_file)
    # print("path")
    # os.makedirs(sys.argv[1])    
    with open(sys.argv[1], "w") as fd:
        json.dump(acc_scores, fd, indent=4)

      
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



