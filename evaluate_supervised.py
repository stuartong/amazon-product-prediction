import pandas as pd
import numpy as np
from sklearn.metrics import plot_precision_recall_curve, f1_score, matthews_corrcoef,confusion_matrix, classification_report,plot_confusion_matrix,recall_score,plot_roc_curve, roc_curve, roc_auc_score, accuracy_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt

def evaluate_model(clf,X_train,X_test,X_val,y_train,y_test,y_val):
    # get predictions for test set
    y_pred = clf.predict(X_test)

    # create & fit dummy classifiers for comparison
    dummy_uniform = DummyClassifier(strategy='uniform',random_state=42).fit(X_train,y_train)
    dummy_mf = DummyClassifier(strategy='most_frequent',random_state=42).fit(X_train,y_train)

    # get dummy predictions
    rand_dev_preds = dummy_uniform.predict(X_test)
    mf_dev_preds = dummy_mf.predict(X_test)

    # Use accuracy to check for overfitting and comp with dummy
    print('Accuracy Scores - Check for Overfit & Compare to Dummy\n')
    train_acc = clf.score(X_train,y_train)
    test_acc = clf.score(X_test,y_test)
    dummy_mf_acc = dummy_mf.score(X_test,y_test)
    dummy_uni_acc = dummy_uniform.score(X_test,y_test)
    print(f'Training Accuracy: {train_acc}')
    print(f'Test Accuracy: {test_acc}')
    print(f'Dummy Most Frequent Accuracy: {dummy_mf_acc}')
    print(f'Dummy Uniform Accuracy: {dummy_uni_acc}')
    print('\n')
    f1_avg = 'weighted'
    print(f'F1 Scores - {f1_avg}')
    test_f1 = f1_score(y_test,y_pred,average=f1_avg)
    dummy_mf_f1 = f1_score(y_test,mf_dev_preds,average=f1_avg)
    dummy_uni_f1 = f1_score(y_test,rand_dev_preds,average=f1_avg)
    print(f'Test F1: {test_f1}')
    print(f'Dummy Most Frequent F1: {dummy_mf_f1}')
    print(f'Dummy Uniform F1: {dummy_uni_f1}')
    print('\n')
    print('Matthews Correlation Coefficient - MCC')
    test_mcc = matthews_corrcoef(y_test,y_pred)
    # dummy_mf_mcc = matthews_corrcoef(y_test,mf_dev_preds)
    dummy_uni_mcc = matthews_corrcoef(y_test,rand_dev_preds)
    print(f'Test MCC: {test_mcc}')
    # print(f'Dummy Most Frequent MCC: {dummy_mf_mcc}')
    print(f'Dummy Uniform MCC: {dummy_uni_mcc}')

    # get classification report 
    print('\nClassification Report - Test Set')
    print(classification_report(y_test,y_pred))

    #display plots
    fig , axes = plt.subplots(1,3,figsize=(24,8))
    axes[0].title.set_text('Confusion Matrix')
    axes[1].title.set_text('ROC Curve')
    axes[2].title.set_text('Precision-Recall Curve')

    cm = plot_confusion_matrix(clf,X_test,y_test,ax=axes[0]);
    roc_curve = plot_roc_curve(clf,X_test,y_test,ax=axes[1]);
    pr_curve = plot_precision_recall_curve(clf,X_test,y_test,ax=axes[2]);

    # cm.plot(ax=axes[0]);
    # roc_curve.plot(ax=axes[1]); 
    # pr_curve.plot(ax=axes[2]);


    return test_f1,test_mcc,test_acc





