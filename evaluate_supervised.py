import pandas as pd
import numpy as np
from sklearn.metrics import plot_precision_recall_curve, f1_score, confusion_matrix, classification_report,plot_confusion_matrix,recall_score,plot_roc_curve, roc_curve, roc_auc_score, accuracy_score
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
    print('\n')
    test_f1 = f1_score(y_test,y_pred,average=f1_avg)
    dummy_mf_f1 = f1_score(y_test,mf_dev_preds,average=f1_avg)
    dummy_uni_f1 = f1_score(y_test,rand_dev_preds,average=f1_avg)
    print(f'Test F1: {test_f1}')
    print(f'Dummy Most Frequent F1: {dummy_mf_f1}')
    print(f'Dummy Uniform F1: {dummy_uni_f1}')


    # get classification report 
    print('\nClassification Report - Test Set')
    print(classification_report(y_test,y_pred))

    # get confusion matrix
    plot_confusion_matrix(clf,X_test,y_test)
    plt.show()

    # get ROC Curve
    plot_roc_curve(clf,X_test,y_test)    
    plt.show()

    # get precision-recall curve
    plot_precision_recall_curve(clf,X_test,y_test)
    plt.show()

    return 





