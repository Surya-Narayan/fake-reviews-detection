#Importing necessary libraries
import string, pandas as pd, numpy as np
from nltk.tokenize import sent_tokenize
from imblearn.under_sampling import NearMiss
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tabulate import tabulate
from scipy.stats import entropy

#Plotting wordclouds of fake and authentic reviews to understand which words occur frequently
def plotwordcloud(table):
    wc = WordCloud(background_color='white', min_font_size=10, width=500, height=500)
    reviews_wc = wc.generate(table[table['label'] == -1]['review'].str.cat(sep=" "))
    plt.figure(figsize=(8, 6))
    plt.imshow(reviews_wc)
    plt.title("Fake Reviews", fontsize=20)
    plt.show()
    reviews_wc = wc.generate(table[table['label'] == 1]['review'].str.cat(sep=" "))
    plt.figure(figsize=(8, 6))
    plt.imshow(reviews_wc)
    plt.title("Authentic Reviews", fontsize=20)
    plt.show()


""" Parameters:
InputTable: A table or dataframe containing the data to be undersampled. This table should have a column named 'label' that indicates the class label for each row.
Return value:
undersampleddata: A new table that contains an equal number of rows for each class label, achieved through random undersampling. This table has the same column structure as the input table. """

def undersample_data(InputTable):
    ClassLength = InputTable.groupby('label').size()

    undersampleddata = InputTable.groupby('label').apply(
        lambda x: x.sample(n=ClassLength.min(), randomseedvalue=573)).reset_index(drop=True)
    return undersampleddata

""" Parameters: 
    InputTable: dataframe containing the raw input data. 

    Return Values: 
    FeatureMatrix: A 2D numpy array containing the preprocessed feature data.
    LabelList: A 1D numpy array containing the preprocessed class labels.
    FeaturesList: A list of strings containing the names of the features
    FeatureMatrix. The order of the feature names corresponds to the order of the columns in FeatureMatrix
"""
def dataPreprocessingFunc(InputTable):
    TLabels = InputTable["label"]
    Removedcolumns = ['user_id', 'prod_id', 'label', 'date', 'review']
    FeatureCols = [col for col in InputTable.columns if col not in Removedcolumns]
    PreprocessedFeatures = InputTable[FeatureCols]
    PreprocessedFeatures = PreprocessedFeatures.apply(lambda x: x / x.abs().max(), axis=0)
    FeaturesList, FeatureMatrix = PreprocessedFeatures.columns.tolist(), PreprocessedFeatures.to_numpy()
    LabelList = np.array([1 if label == -1 else 0 for label in TLabels])
    return FeatureMatrix, LabelList, FeaturesList


""" Parameters: 
    InputTable: A table or dataframe containing the raw input data.
    Return Values:
    FeatureMatrix: A 2D numpy array containing the preprocessed feature data.
"""
def DataFramePreprocessed(InputTable):
    Removedcolumns = ['user_id', 'prod_id', 'label', 'date', 'review']
    FeatureCols = [col for col in InputTable.columns if col not in Removedcolumns]
    PreprocessedFeatures = InputTable[FeatureCols]
    PreprocessedFeatures = PreprocessedFeatures.apply(lambda x: x / x.abs().max(), axis=0)
    return PreprocessedFeatures


""" 
This function calculates the evaluation metrics - Precision, Recall, Specificity and F1 Score
"""
def calculatemetrics(true_labels, predicted_labels, evaluation_type):
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    specificity = tn / (tn + fp)

    MetricLabels = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score']
    MetricVals = [accuracy_score(true_labels, predicted_labels),
                     precision_score(true_labels, predicted_labels),
                     recall_score(true_labels, predicted_labels),
                     specificity,
                     f1_score(true_labels, predicted_labels)]

    DfMetric = pd.DataFrame(list(zip(MetricLabels, MetricVals)), columns=["metric_type", "value"])
    DfMetric.to_csv(f"./EVALUATIONS/Metrics_{evaluation_type}.csv", header=False, index=False)

    
    ConfusionMatrixValues = {'Predicted Positive': [tp, fn],
                             'Predicted Negative': [fp, tn]}
    ConfusionmatrixIndex = ['Actual Positive', 'Actual Negative']

    DFConfusionMatrix = pd.DataFrame(ConfusionMatrixValues, index=ConfusionmatrixIndex)
   
    print("\nConfusion Matrix:")
    print(tabulate(DFConfusionMatrix, headers='keys', tablefmt='psql'))

  
    print("\nMetric Values:")
    print(tabulate(DfMetric, headers='keys', tablefmt='psql'))


""" Parameters:
    group: A Pandas group object as input
    Return value:
    A Pandas Series object with the same length as the input group. """
def SingletonFunc(group):
    groupShape = group.shape[0]
    return pd.Series([1 if groupShape == 1 else 0] * groupShape, index=group.index)


def DataViewFunc(InputTable):
    InputTable['singleton'] = InputTable.groupby(['user_id', 'date']).apply(SingletonFunc).reset_index(
        drop=True)
    return InputTable[['singleton']]

def textDatafunc(table):
    TableStats = {"CapLetterRatio": [], "CapitalWordRatio": [
    ], "FirstPersRatio": [], "ExclRatio": []}  # , "sentiment":[]
    SetFirstPerson = set(["i", "mine", "my", "me", "we", "our", "us", "ourselves", "ours"])

    for i, row in table.iterrows():
        sentences = sent_tokenize(row["review"])
        ExclamationCount, ReviewWordCount  = 0
        CapLetterCount, CapWordCount, FirstPersonCount  = 0
        for sentence in sentences:
            if sentence[-1] == "!":
                ExclamationCount += 1
            sentence = sentence.translate(
                str.maketrans('', '', string.punctuation))
            sentence = sentence.split(" ")
            ReviewWordCount += len(sentence)

            for word in sentence:
                if word.isupper():
                    CapWordCount += 1
                if word.lower() in SetFirstPerson:
                    FirstPersonCount += 1
                for w in word:
                    if w.isupper():
                        CapLetterCount += 1
                        break

        ExclRatio = ExclamationCount / len(sentences)
        CapLetterRatio = CapLetterCount / ReviewWordCount
        CapitalWordRatio = CapWordCount / ReviewWordCount
        FirstPersRatio = FirstPersonCount / ReviewWordCount
        TableStats["ExclRatio"].append(ExclRatio)
        TableStats["CapLetterRatio"].append(CapLetterRatio)
        TableStats["CapitalWordRatio"].append(CapitalWordRatio)
        TableStats["FirstPersRatio"].append(FirstPersRatio)

    DFTextStats = pd.DataFrame.from_dict(TableStats)
    return DFTextStats


def FuncRatingFeatures(table):
   
    avgProdIDRating = table.groupby('prod_id')['rating'].mean().reset_index(name='prod_avg')
    table = table.merge(avgProdIDRating, on='prod_id', validate='m:1')
    table['AverageDeviationFromEntityAvg'] = (table['rating'] - table['prod_avg']).abs()

    CountUserIDRation = table.groupby(['user_id', 'rating']).size().reset_index(name='count')
    UserCounts = CountUserIDRation.groupby('user_id')['count'].sum().reset_index(name='total_count')
    CountUserIDRation = CountUserIDRation.merge(UserCounts, on='user_id', validate='m:1')
    CountUserIDRation['prob'] = CountUserIDRation['count'] / CountUserIDRation['total_count']
    UserRatingEntropy = CountUserIDRation.groupby('user_id').apply(lambda x: entropy(x['prob'])).reset_index(
        name='RatingEntropy')
    table = table.merge(UserRatingEntropy, on='user_id', validate='m:1')
 
    avg_user_rating = table.groupby('user_id')['rating'].mean().reset_index(name='user_avg')
    table = table.merge(avg_user_rating, on='user_id', validate='m:1')
    table['RatingVariance'] = (table['rating'] - table['user_avg']) ** 2

    return table[['AverageDeviationFromEntityAvg', 'RatingEntropy', 'RatingVariance']]


def FuncBehavioralFeatures(table):

    reviewCount = table.groupby(['user_id', 'date']).size().reset_index(name='count')
    mnr = reviewCount.groupby('user_id')['count'].max().reset_index(name='MaxReviews')
    table = table.merge(mnr, on='user_id', validate='m:1')
   
    total_reviews = table.groupby('user_id')['rating'].count().reset_index(name='total')
    PosReviews = table[table['rating'] >= 4].groupby('user_id')['rating'].count().reset_index(name='pos')
    NegReviews = table[table['rating'] <= 2].groupby('user_id')['rating'].count().reset_index(name='neg')
    MergedTotal = total_reviews.merge(PosReviews, on='user_id', how='left').fillna(0).merge(NegReviews,
                                                                                                    on='user_id',
                                                                                                    how='left').fillna(
        0)
    MergedTotal['PerPosReviews'] = MergedTotal['pos'] / MergedTotal['total']
    MergedTotal['PerNegReviews'] = MergedTotal['neg'] / MergedTotal['total']
    table = table.merge(MergedTotal[['user_id', 'PerPosReviews', 'PerNegReviews']], on='user_id', validate='m:1')

    table['Reviewlength'] = table['review'].str.split().str.len()
    ReviewLengthAvg = table.groupby('user_id')['Reviewlength'].mean().reset_index(name='RL')
    table = table.merge(ReviewLengthAvg, on='user_id', validate='m:1')

   
    avgProdIDRating = table.groupby('prod_id')['rating'].mean().reset_index(name='avg')
    table = table.merge(avgProdIDRating, on='prod_id', validate='m:1')
    table['RatingDev'] = (table['rating'] - table['avg']).abs()
    ReviewerDev = table.groupby('user_id')['RatingDev'].mean().reset_index(name='ReviewerDev')
    table = table.merge(ReviewerDev, on='user_id', validate='m:1')

    return table[['MaxReviews', 'PerPosReviews', 'PerNegReviews', 'RL', 'RatingDev', 'ReviewerDev']]


def funcTemporalFeat(data):
    data['date'] = pd.to_datetime(data['date'])
    DataSelected = data.loc[:, ['user_id', 'date', 'rating']]
    DataSelected.sort_values(by=['date'], inplace=True)
    ActivityTimeTable = DataSelected[['user_id', 'date']].groupby(['user_id']).agg(
        first_date=pd.NamedAgg(column='date', aggfunc='min'),
        last_date=pd.NamedAgg(column='date', aggfunc='max')
    )
    ActivityTimeTable['ActivityTime'] = (
            (ActivityTimeTable['last_date'] - ActivityTimeTable['first_date']) / np.timedelta64(1, 'D')).astype(int)

  
    TempData = data
    TempData['date'] = pd.to_datetime(TempData['date'])
    TempData = TempData[['user_id', 'date', 'rating']].groupby(['user_id', 'date']).agg(
        RatingMax=pd.NamedAgg(column='rating', aggfunc='max')
    )

   
    DataSelected['previous_date'] = DataSelected.groupby('user_id')['date'].shift()
    DataSelected['DateEntropy'] = DataSelected['date'] - DataSelected['previous_date']
    DataSelected.replace({pd.NaT: '0 day'}, inplace=True)
    DataSelected['DateEntropy'] = (DataSelected['DateEntropy'] / np.timedelta64(1, 'D')).astype(int)

    
    DataSelected['original_index'] = DataSelected.index
    temp_data3 = DataSelected[['user_id', 'date']].groupby(['user_id']).agg(
        avg_date=pd.NamedAgg(column='date', aggfunc='mean')
    )
    DataSelected = pd.merge(DataSelected, temp_data3, on='user_id', how='left')
    DataSelected['DateVar'] = abs(((DataSelected['date'] - DataSelected['avg_date']) / np.timedelta64(1, 'D'))) ** 2
    DataSelected.set_index('original_index')

   
    data = data.loc[:, ['user_id', 'date']]
    data['date'] = pd.to_datetime(data['date'])
    data = pd.merge(data, ActivityTimeTable, on='user_id', how='left')
    data = pd.merge(data, TempData, on=['user_id', 'date'], how='left')
    data = pd.merge(data, DataSelected, left_on=data.index, right_on='original_index')

    return data[['ActivityTime', 'RatingMax', 'DateEntropy', 'DateVar']]

def ReviewerFunc(table):
  
    GroupedData = table.groupby(['prod_id', 'date'])
  
    density = GroupedData.size().reset_index(name='density')
    table = table.merge(density, on=['prod_id', 'date'], validate='m:1')
  
    avg_date = GroupedData['rating'].mean().reset_index(name='avg_date')
    table = table.merge(avg_date, on=['prod_id', 'date'], validate='m:1')
   
    avg = table.groupby('prod_id')['rating'].mean().reset_index(name='avg')
    table = table.merge(avg, on='prod_id', validate='m:1')
    table['DevFromMean'] = (table['rating'] - table['avg_date']).abs()
    table['MeanRatingDev'] = (table['avg_date'] - table['avg']).abs()

    return table[['density', 'MeanRatingDev', 'DevFromMean']]
