import sklearn
import numpy as np
import os.path
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from feature_extraction import *


def svm_feature_selection(train_features, train_labels, test_features, test_labels, feature_names, type=""):
    """
    Trains svm model with inputted features. Visualizes and returns top n coeff feature names
    """
    # check if trained model already exists
    root = os.getcwd()
    filepath = os.path.join(root, 'CODE', 'models', 'lin_svm_v2.joblib')
    if not os.path.exists(filepath):
        train_lin_model(train_features, train_labels, filepath)

    # uncomment this line to predict classification for test set
    # y_pred = clf.predict(X_test)
    clf = load(filepath)

    test_pred = clf.predict(test_features)
    evaluate(test_labels, test_pred, type)

    return feature_selection(clf, feature_names)


def train_lin_model(features, labels, filepath):
    clf = svm.SVC(kernel='linear')
    clf.fit(features, labels)

    # save svm model
    dump(clf, filepath)


import numpy as np
import matplotlib.pyplot as plt
import os

def feature_selection(classifier, feature_names, n=10):
    coefficients = classifier.coef_.ravel()

    # Get top n/2 positive and negative coefficients
    top_positive_coeffs = np.argsort(coefficients)[-int(n/2):]
    top_negative_coeffs = np.argsort(coefficients)[:int(n/2)]
    top_coeffs = np.hstack([top_negative_coeffs, top_positive_coeffs])

    # Create plot
    plt.figure(figsize=(10, 5))
    colors = ['red' if coeff < 0 else 'blue' for coeff in coefficients[top_coeffs]]
    y_pos = np.arange(n)
    plt.barh(y_pos, coefficients[top_coeffs], color=colors)
    feature_names = np.array(feature_names)
    plt.yticks(y_pos, feature_names[top_coeffs], rotation=0)
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/SVM_feature_importance.png')
    plt.show()
    return [feature_names[i] for i in top_coeffs], [feature_names[top_negative_coeffs[0]], feature_names[top_positive_coeffs[-1]]]


