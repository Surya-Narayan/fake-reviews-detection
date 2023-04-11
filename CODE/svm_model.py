import numpy as np
import os.path
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from feature_extraction import *

def feature_selection_plot(weights, sorted_indices, features, n):
    """
    This function creates a horizontal bar plot of the feature importances (weights) of the top 'n' features sorted by their indices.
    The bar colors represent the sign of the weights (red for negative weights and blue for positive weights).
    
    Parameters:
    weights (array-like): The weights of the features.
    sorted_indices (array-like): The indices of the features sorted by their importance.
    features (list): A list of feature names for the dataset.
    n (int): The number of top features to be plotted.
    """
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

def svm_feature_selection(train_data, train_targets, test_data, test_targets, feature_names, type=""):
    """
    This function trains a linear SVM model with the given training data and targets. If a saved model already exists, it loads the model from disk.
    The function evaluates the model using the test data and targets, and returns the selected features.
    
    Parameters:
    train_data (array-like): The training dataset features.
    train_targets (array-like): The training dataset labels.
    test_data (array-like): The testing dataset features.
    test_targets (array-like): The testing dataset labels.
    feature_names (list): A list of feature names for the dataset.
    type (str, optional): A string representing the model type. Default is an empty string.
    
    Returns:
    list: A list containing the selected features.
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
    """
    This function trains a linear SVM model with the given data and targets, and saves the model to the specified file path.
    
    Parameters:
    data (array-like): The dataset features.
    targets (array-like): The dataset labels.
    model_path (str): The file path where the trained model should be saved.
    """    
    linear_clf = svm.SVC(kernel='linear')
    linear_clf.fit(data, targets)

    # save svm model
    dump(linear_clf, model_path)


import numpy as np
import matplotlib.pyplot as plt
import os

def feature_selection(model, features, n=10):
    """
    This function selects the top 'n' features based on the weights of the linear SVM model.
    The function plots the feature importances of the selected features and returns the feature names
    and the most negatively and positively weighted features.
    
    Parameters:
    model (sklearn.svm.SVC): The trained linear SVM model.
    features (list): A list of feature names for the dataset.
    n (int, optional): The number of top features to be selected. Default is 10.
    
    Returns:
    tuple: A tuple containing the following elements:
        - A list of the selected top 'n' feature names.
        - A list containing the most negatively and positively weighted feature names.
    """

    weights = model.coef_.ravel()
    # Get the indices of the top n/2 positive and negative weights
    pos_indices = np.argpartition(weights, -int(n/2))[-int(n/2):]
    neg_indices = np.argpartition(weights, int(n/2))[:int(n/2)]
    top_indices = np.concatenate((neg_indices, pos_indices))

    # Sort the indices
    sorted_indices = top_indices[np.argsort(weights[top_indices])]
    feature_selection_plot(weights, sorted_indices, features, n)

    return [features[i] for i in sorted_indices], [features[neg_indices[0]], features[pos_indices[-1]]]