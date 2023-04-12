import os.path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from scipy.stats import entropy
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from imblearn.under_sampling import NearMiss
from nltk.tokenize import sent_tokenize
from statistics import mean
import matplotlib as mpl
from FeatureEngineering import *
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import joblib
import plotly.express as px


"""

RANDOM FOREST BEGINS BELOW


"""


def compare_rf_plot(num_estimators, train_scores, test_scores):
    """
    This function creates a plot to visualize the performance of the RandomForestClassifier model with different hyperparameter configurations.
    It takes as input the number of estimators, train scores, and test scores, and generates a plot showing the Area Under the Curve (AUC) scores for both the train and test datasets.
    The plot is saved as an image in the 'EVALUATIONS' directory.
    """
    plt.plot(num_estimators, train_scores, 'g', label="Train AUC")
    plt.plot(num_estimators, test_scores, 'r', label="Test AUC")
    plt.legend(loc='lower right')
    plt.ylabel('Area Under Curve Score')
    plt.xlabel('Number of Estimators')
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/RF_auc_scores.png')
    plt.show()

def feature_importance_plot(importance_df):
    """
    This function creates a bar plot to visualize the feature importance of a model, given a DataFrame with feature names and their corresponding Gini-importance values.
    The plot displays the importance of each feature on the y-axis and the feature names on the x-axis.
    The plot is saved as an image in the 'EVALUATIONS' directory.
    """
    sns.set(style="darkgrid", color_codes=True, font_scale=2)
    fig, ax = plt.subplots()
    fig.set_size_inches(50, 25)
    sns.barplot(x=importance_df['Gini-Importance'], y=importance_df['Feature_Name'], data=importance_df, color='magenta')
    plt.xlabel('Importance', fontsize=18)
    plt.ylabel('Feature_Name', fontsize=18)
    plt.title('Feature Importance', fontsize=22, weight='bold')
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/RF_feature_importance.png')
    plt.show()

# perform undersampling
def train_rf(training_features, testing_features, training_labels, testing_labels, feature_name_list, type="rf"):
    """
    This function trains a RandomForestClassifier model with the given training features and labels. If a saved model already exists, it loads the model from disk.
    The function evaluates the model using the testing features and labels, and returns the feature importances.
    
    Parameters:
    training_features (array-like): The training dataset features.
    testing_features (array-like): The testing dataset features.
    training_labels (array-like): The training dataset labels.
    testing_labels (array-like): The testing dataset labels.
    feature_name_list (list): A list of feature names for the dataset.
    type (str, optional): A string representing the model type. Default is "rf" (Random Forest).
    
    Returns:
    DataFrame: A DataFrame containing the feature importances.
    """    
    project_directory = os.getcwd()
    rf_model_file_path = os.path.join(project_directory, 'CODE', 'models', 'rf_v1.joblib')
    
    def create_rf_classifier():
        rf_classifier = RandomForestClassifier(n_estimators=56, max_depth=10)
        rf_classifier.fit(training_features, training_labels)
        return rf_classifier
    
    if not os.path.exists(rf_model_file_path):
        rf_classifier_model = create_rf_classifier()
        joblib.dump(rf_classifier_model, rf_model_file_path)
    else:
        with open(rf_model_file_path, 'rb') as classifier_file:
            rf_classifier_model = joblib.load(classifier_file)
        
    predictions_from_classifier = rf_classifier_model.predict(testing_features)
    calculatemetrics(testing_labels, predictions_from_classifier, type)

    return feature_importance(rf_classifier_model, feature_name_list)

def feature_importance(trained_model, feature_names):
    """
    This function calculates the feature importances for a trained model and generates a bar plot of the feature importances.
    The function also returns a list of the top 10 features with the highest importance.
    
    Parameters:
    trained_model (fitted model object): A trained model object (e.g., RandomForestClassifier).
    feature_names (list): A list of feature names for the dataset.
    
    Returns:
    list: A list containing the top 10 features with the highest Gini-importance.
    """
    feature_importance_pairs = list(zip(feature_names, trained_model.feature_importances_))
    sorted_feature_importance = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
    
    importance_df = pd.DataFrame(sorted_feature_importance, columns=['Feature_Name', 'Gini-Importance'])
    feature_importance_plot(importance_df)

    top_features = [importance_df['Feature_Name'][i] for i in range(10)]
    return top_features

def PCAVisualization(input_data, target_labels):
    """
    This function applies Principal Component Analysis (PCA) to reduce the input data to three dimensions and creates a 3D scatter plot of the transformed data.
    The plot is colored based on the target labels and saved as an image in the 'EVALUATIONS' directory.
    
    Parameters:
    input_data (array-like): The input dataset to be transformed using PCA.
    target_labels (array-like): The target labels corresponding to the input dataset.
    """
    pca_reducer = PCA(n_components=3)
    transformed_data = pca_reducer.fit_transform(input_data)
    df_transformed_data = pd.DataFrame(transformed_data, columns=['PC1', 'PC2', 'PC3'])
    df_transformed_data['labels'] = target_labels

    fig = px.scatter_3d(df_transformed_data, x='PC1', y='PC2', z='PC3', color='labels')
    fig.write_image("./EVALUATIONS/RF_PCA_3D_Visualization.png")
    fig.show()

def randomForest(x_train_data, y_train_data, x_test_data, y_test_data):
    """
    This function trains a RandomForestClassifier model with different hyperparameter configurations and compares their performance.
    The hyperparameters used are the number of estimators and maximum tree depth, and the performance is evaluated using Area Under the Curve (AUC) scores.
    The function calls 'compare_rf_plot' to generate a plot of the AUC scores for the different hyperparameter configurations.
    
    Parameters:
    x_train_data (array-like): The training dataset features.
    y_train_data (array-like): The training dataset labels.
    x_test_data (array-like): The testing dataset features.
    y_test_data (array-like): The testing dataset labels.
    """
    num_estimators = [2, 5, 10, 20, 40, 60, 80, 120, 150]
    max_tree_depths = [1, 3, 4, 6, 8, 11, 13, 18, 25]
    train_scores = []
    test_scores = []

    for estimator, depth in zip(num_estimators, max_tree_depths):
        rf_classifier = RandomForestClassifier(n_estimators=estimator, max_depth=depth)
        rf_classifier.fit(x_train_data, y_train_data)
        
        train_pred_prob = rf_classifier.predict_proba(x_train_data)[:, 1]
        test_pred_prob = rf_classifier.predict_proba(x_test_data)[:, 1]

        train_auc = roc_auc_score(y_train_data, train_pred_prob)
        test_auc = roc_auc_score(y_test_data, test_pred_prob)

        train_scores.append(train_auc)
        test_scores.append(test_auc)

    compare_rf_plot(num_estimators, train_scores, test_scores)





    

"""

SVM MODEL BEGINS BELOW 


"""


import numpy as np
import os.path
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from FeatureEngineering import *

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
    plt.yticks(y_positions, features[sorted_indices])
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
    calculatemetrics(test_targets, test_predictions, type)
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
