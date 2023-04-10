import numpy as np
import os.path
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from feature_extraction import *


def svm_feature_selection(train_data, train_targets, test_data, test_targets, feature_names, type=""):
    """
    Trains svm model with inputted features. Visualizes and returns top n coeff feature names
    """
    # check if trained model already exists
    root_dir = os.getcwd()
    model_path = os.path.join(root_dir, 'CODE', 'models', 'lin_svm_v2.joblib')
    if not os.path.exists(model_path):
        fit_linear_model(train_data, train_targets, model_path)

    # uncomment this line to predict classification for test set
    # y_pred = clf.predict(X_test)
    clf = load(model_path)

    test_predictions = clf.predict(test_data)
    evaluate(test_targets, test_predictions, type)
    return feature_selection(clf, feature_names)

def fit_linear_model(data, targets, model_path):
    linear_clf = svm.SVC(kernel='linear')
    linear_clf.fit(data, targets)

    # save svm model
    dump(linear_clf, model_path)


import numpy as np
import matplotlib.pyplot as plt
import os

def feature_selection(model, features, n=10):
    weights = model.coef_.ravel()

    # Get the indices of the top n/2 positive and negative weights
    pos_indices = np.argpartition(weights, -int(n/2))[-int(n/2):]
    neg_indices = np.argpartition(weights, int(n/2))[:int(n/2)]
    top_indices = np.concatenate((neg_indices, pos_indices))

    # Sort the indices
    sorted_indices = top_indices[np.argsort(weights[top_indices])]

    # Create plot
    plt.figure(figsize=(10, 5))
    bar_colors = ['red' if w < 0 else 'blue' for w in weights[sorted_indices]]
    y_positions = np.arange(n)
    plt.barh(y_positions, weights[sorted_indices], color=bar_colors)
    features = np.array(features)
    plt.yticks(y_positions, features[sorted_indices], rotation=0)
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/SVM_feature_importance.png')
    plt.show()

    return [features[i] for i in sorted_indices], [features[neg_indices[0]], features[pos_indices[-1]]]
