#Importing necessary libraries
import pandas as pd, nltk
from pathlib import Path 
from DeepLearning import *
from rf_svm_models import *
from FeatureEngineering import *


def main():
    maindirectory = os.getcwd()
    rightdirectory = False
    for file in os.listdir(maindirectory):
        direc = os.path.join(maindirectory, file)
        if os.path.isdir(direc):
            if 'CODE' in direc.split('\\')[-1]:
                rightdirectory = True
    if not rightdirectory:
        print("Error: Please run python3 ./CODE/main.py")
        return
    
    #Fetching the YelpDataset
    FilepathSampled = Path(f'{os.getcwd()}/DATA/FinalDataset.csv')
    if not os.path.exists(FilepathSampled):
        #File location for storing the feature engineered attributes
        filepath = Path(f'{os.getcwd()}/DATA/FeatureEngineeredAttributes.csv')
        if not os.path.exists(filepath):
            nltk.download('punkt')

            FilePath = f'{os.getcwd()}/DATA/YelpDataSetCSV'
            ReviewAttributesColumns = ["user_id", "prod_id", "date", "review"]
            ReviewAttributesData = pd.read_csv(FilePath+"/ReviewAttributes.csv", names=ReviewAttributesColumns)
            DateRatingcolumns = ["user_id", "prod_id", "rating", "label", "date"]
            DateRatingData = pd.read_csv(FilePath+"/DateRatingAttributes.csv", names=DateRatingcolumns)
            #Creating a consolidated table by combining the features in ReviewAttributes.csv and DateRatingAttributes.csv
            ConsolidatedTable = pd.concat([DateRatingData, ReviewAttributesData["review"]], axis=1).dropna()
            ConsolidatedTable=pd.concat([ConsolidatedTable, DataViewFunc(ConsolidatedTable), textDatafunc(ConsolidatedTable), ReviewerFunc(ConsolidatedTable), FuncBehavioralFeatures(ConsolidatedTable), FuncRatingFeatures(ConsolidatedTable),
                             funcTemporalFeat(ConsolidatedTable)], axis=1)

            os.makedirs(filepath, exist_ok=True)  
            ConsolidatedTable.to_csv(filepath, index=False)
    
        ConsolidatedTable = pd.read_csv(filepath)

        os.makedirs(FilepathSampled, exist_ok=True)
        #undersampling the data to remove any bias
        TableUndersampled = undersample_data(ConsolidatedTable)
        TableUndersampled.to_csv(FilepathSampled, index=False)

    #Final Samples upon Preprocessing
    FinalSamples = pd.read_csv(FilepathSampled)
    features, labels, FeatureName = dataPreprocessingFunc(FinalSamples)
    PreprocessedFeatures = DataFramePreprocessed(FinalSamples)
    
     
    #Splitting the data
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,random_state=573)
    
    # Calling the function to plot wordcloud
    plotwordcloud(FinalSamples)
    randomForest(train_features,train_labels,test_features,test_labels)
   
    SelectedRFFeatues = train_rf(train_features, test_features, train_labels, test_labels,FeatureName, type="rf")
    print(SelectedRFFeatues)
    PCAVisualization(train_features,train_labels)

    SelectedSVMFeatures, SVMTop2Features = svm_feature_selection(train_features, train_labels, test_features, test_labels, FeatureName, type="lin_svm")
    print(SelectedSVMFeatures)
    

    FeaturesSVM = PreprocessedFeatures[SelectedSVMFeatures]
    FeaturesRF = PreprocessedFeatures[SelectedRFFeatues]
    FeaturesIDS = range(len(SelectedSVMFeatures))
    RFTop2Features = SelectedRFFeatues[:2]
    SVMTop2Features = SelectedSVMFeatures[0] + SelectedSVMFeatures[-1]
    #Invoking the DL Model
    print("training DL model with all features")
    DeepLearningModel(PreprocessedFeatures, labels, FeaturesIDS, RFTop2Features , n_epoches = 100, batch_size = 256, type="DL_all")

    print("trainin DL model with top features from RF features")
    DeepLearningModel(FeaturesRF, labels, FeaturesIDS, RFTop2Features , n_epoches = 100, batch_size = 256, type="DL_rf")

    print("trainin DL model with top features from SVM features")
    DeepLearningModel(FeaturesSVM, labels, FeaturesIDS, RFTop2Features , n_epoches = 100, batch_size = 256, type="DL_svm")


if __name__ == "__main__":
    main()
