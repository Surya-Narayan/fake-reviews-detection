from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import nltk
#from feature_extraction import *
import pandas as pd
import numpy as np
import string
import re
import collections
from statistics import mean
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import entropy
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import Normalizer
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import plotly.express as px
import os.path
from feature_extraction import *


# perform undersampling

# def train_rf(train_features, test_features, train_labels, test_labels,feature_names, type=""):

#     root = os.getcwd()
#     filepath = os.path.join(root, 'CODE', 'models', 'rf_v1.joblib')
#     if not os.path.exists(filepath):
#         clf = RandomForestClassifier(n_estimators=64,max_depth=10)
#         clf.fit(train_features, train_labels)
#         joblib.dump(clf, filepath)
#     clf = joblib.load(filepath)    
#     y_pred=clf.predict(test_features)
#     evaluate(test_labels, y_pred, type)

#     return feature_importance(clf,feature_names)

def train_rf(train_data, validation_data, train_target, validation_target, feature_list, type="rf"):
    root_dir = os.getcwd()
    model_path = os.path.join(root_dir, 'CODE', 'models', 'rf_v1.joblib')
    if not os.path.exists(model_path):
        random_forest = RandomForestClassifier(n_estimators=64, max_depth=10)
        random_forest.fit(train_data, train_target)
        joblib.dump(random_forest, model_path)
    random_forest = joblib.load(model_path)
    predicted_values = random_forest.predict(validation_data)
    evaluate(validation_target, predicted_values, type)

    return feature_importance(random_forest, feature_list)


# def feature_importance(clf,feature_names):
    

#     feats = {}
#     for feature, importance in zip(feature_names, clf.feature_importances_):
#         feats[feature] = importance
#     importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
#     importances = importances.sort_values(by='Gini-Importance', ascending=False)
#     importances = importances.reset_index()
#     importances = importances.rename(columns={'index': 'Features'})
#     sns.set(font_scale = 5)
#     sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
#     fig, ax = plt.subplots()
#     fig.set_size_inches(30,15)
#     sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
#     plt.xlabel('Importance', fontsize=25, weight = 'bold')
#     plt.ylabel('Features', fontsize=25, weight = 'bold')
#     plt.title('Feature Importance', fontsize=25, weight = 'bold')
#     plt.savefig(f'{os.getcwd()}/EVALUATIONS/RF_feature_importance.png')
#     plt.show()
    
#     return [importances['Features'][i] for i in range(10)]

def feature_importance(model, feature_list):
    feature_significance_dict = {}
    for feat, significance in zip(feature_list, model.feature_importances_):
        feature_significance_dict[feat] = significance
    significance_df = pd.DataFrame.from_dict(feature_significance_dict, orient='index').rename(columns={0: 'Gini-Significance'})
    significance_df = significance_df.sort_values(by='Gini-Significance', ascending=False)
    significance_df = significance_df.reset_index()
    significance_df = significance_df.rename(columns={'index': 'Feature_Name'})
    sns.set(font_scale=5)
    sns.set(style="darkgrid", color_codes=True, font_scale=1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 15)
    sns.barplot(x=significance_df['Gini-Significance'], y=significance_df['Feature_Name'], data=significance_df, color='magenta')
    plt.xlabel('Significance', fontsize=25, weight='bold')
    plt.ylabel('Feature_Name', fontsize=25, weight='bold')
    plt.title('Feature Significance', fontsize=25, weight='bold')
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/RF_feature_significance.png')
    plt.show()

    result = [significance_df['Feature_Name'][i] for i in range(10)]
    return result
    
# def pca_visualization(train_features,labels):
#     pca = PCA(n_components=2)
#     components = pca.fit_transform(train_features)
#     fig = px.scatter(components, x=0, y=1, color=labels)
#     fig.show()

def pca_visualization(train_data, target_labels):
    reduced_pca = PCA(n_components=3)
    principal_components = reduced_pca.fit_transform(train_data)
    scatter_plot = px.scatter_3d(principal_components, x=0, y=1, z=2, color=target_labels)
    scatter_plot.write_image("./EVALUATIONS/RF_PCA_Visualization.png")
    scatter_plot.show()

def compare_rf(x_train_data, y_train_data, x_test_data, y_test_data):
    num_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
    max_tree_depths = [1, 2, 3, 4, 8, 9, 10, 16, 32]
    train_scores = []
    test_scores = []
    
    for idx, estimator in enumerate(num_estimators):
        rf_classifier = RandomForestClassifier(n_estimators=estimator, max_depth=max_tree_depths[idx])
        rf_classifier.fit(x_train_data, y_train_data)
        train_predictions = rf_classifier.predict(x_train_data)
        fpr_train, tpr_train, _ = roc_curve(y_train_data, train_predictions)
        train_auc = auc(fpr_train, tpr_train)
        train_scores.append(train_auc)
        
        test_predictions = rf_classifier.predict(x_test_data)
        fpr_test, tpr_test, _ = roc_curve(y_test_data, test_predictions)
        test_auc = auc(fpr_test, tpr_test)
        test_scores.append(test_auc)
        
    train_line, = plt.plot(num_estimators, train_scores, 'g', label="Train AUC")
    test_line, = plt.plot(num_estimators, test_scores, 'r', label="Test AUC")
    plt.legend(handler_map={train_line: HandlerLine2D(numpoints=2)})
    plt.ylabel('Area Under Curve Score')
    plt.xlabel('Number of Estimators')
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/RF_accuracy_scores.png')
    plt.show()