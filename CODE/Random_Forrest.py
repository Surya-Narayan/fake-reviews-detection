import os.path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.stats import entropy
import seaborn as sns
import re
import collections
import string
from sklearn.metrics import roc_curve, auc
from imblearn.under_sampling import NearMiss
from nltk.tokenize import sent_tokenize
from statistics import mean
import matplotlib as mpl
from feature_extraction import *
from sklearn.decomposition import PCA
import nltk
from sklearn.metrics.pairwise import pairwise_distances
import joblib
import plotly.express as px


# perform undersampling
def train_rf(training_features, testing_features, training_labels, testing_labels, feature_name_list, type="rf"):
    project_directory = os.getcwd()
    rf_model_file_path = os.path.join(project_directory, 'CODE', 'models', 'rf_v1.joblib')
    
    def create_rf_classifier():
        rf_classifier = RandomForestClassifier(n_estimators=64, max_depth=10)
        rf_classifier.fit(training_features, training_labels)
        return rf_classifier
    
    if not os.path.exists(rf_model_file_path):
        rf_classifier_model = create_rf_classifier()
        joblib.dump(rf_classifier_model, rf_model_file_path)
    else:
        with open(rf_model_file_path, 'rb') as classifier_file:
            rf_classifier_model = joblib.load(classifier_file)
        
    predictions_from_classifier = rf_classifier_model.predict(testing_features)
    evaluate(testing_labels, predictions_from_classifier, type)

    return feature_importance(rf_classifier_model, feature_name_list)

def feature_importance(trained_model, feature_names):
    feature_importance_pairs = list(zip(feature_names, trained_model.feature_importances_))
    sorted_feature_importance = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
    
    importance_df = pd.DataFrame(sorted_feature_importance, columns=['Feature_Name', 'Gini-Importance'])
    
    sns.set(font_scale=5)
    sns.set(style="darkgrid", color_codes=True, font_scale=1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 15)
    sns.barplot(x=importance_df['Gini-Importance'], y=importance_df['Feature_Name'], data=importance_df, color='blue')
    plt.xlabel('Importance', fontsize=25, weight='bold')
    plt.ylabel('Feature_Name', fontsize=25, weight='bold')
    plt.title('Feature Importance', fontsize=25, weight='bold')
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/RF_feature_importance.png')
    plt.show()

    top_features = [importance_df['Feature_Name'][i] for i in range(10)]
    return top_features

def pca_visualization(input_data, target_labels):
    pca_reducer = PCA(n_components=3)
    transformed_data = pca_reducer.fit_transform(input_data)
    df_transformed_data = pd.DataFrame(transformed_data, columns=['PC1', 'PC2', 'PC3'])
    df_transformed_data['labels'] = target_labels

    fig = px.scatter_3d(df_transformed_data, x='PC1', y='PC2', z='PC3', color='labels')
    fig.write_image("./EVALUATIONS/RF_PCA_3D_Visualization.png")
    fig.show()

def compare_rf(x_train_data, y_train_data, x_test_data, y_test_data):
    num_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
    max_tree_depths = [1, 2, 3, 4, 8, 9, 10, 16, 32]
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

    plt.plot(num_estimators, train_scores, 'g', label="Train AUC")
    plt.plot(num_estimators, test_scores, 'r', label="Test AUC")
    plt.legend(loc='lower right')
    plt.ylabel('Area Under Curve Score')
    plt.xlabel('Number of Estimators')
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/RF_auc_scores.png')
    plt.show()