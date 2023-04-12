#Importing necessary libraries
import torch, csv, os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


# setting the seed value
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 
np.random.seed(42000)
torch.manual_seed(42000)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42000)

#Importing additional libraries for ploting visualizations
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from FeatureEngineering import *

class YelpDataset(Dataset):
    def __init__(self, features,labels,idfeature, mode='train',):
        self.mode = mode
        labels = np.array(labels)
        #Creating a Numpy array of Yelp Dataset
        loadeddata = np.array(features).astype(float)
        LoadedFeatures = idfeature
        if mode == 'test':
            loadeddata = loadeddata[:, LoadedFeatures]
            self.loadeddata = torch.FloatTensor(loadeddata)
        else:
            targetlabels = labels
            print(len(loadeddata[0]))
            loadeddata = loadeddata[:, LoadedFeatures]
          
            if mode == 'train':
                index = [i for i in range(len(loadeddata)) if i % 20 != 0]
            elif mode == 'dev':
                index = [i for i in range(len(loadeddata)) if i % 20 == 0]
            
          
            self.loadeddata = torch.FloatTensor(loadeddata[index])
            self.targetlabels = torch.FloatTensor(targetlabels[index])

        self.loadeddata = \
            (self.loadeddata- self.loadeddata.mean(dim=0, keepdim=True)) \
            / self.loadeddata.std(dim=0, keepdim=True)

        self.dim = self.loadeddata.shape[1]

        print('Finished reading the {} set of Yelp Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.loadeddata), self.dim))

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.loadeddata[index], self.targetlabels[index]
        else:
            return self.loadeddata[index]

    def __len__(self):
        return len(self.loadeddata)


#Function that returns the string cuda if available for use
def DeviceTypefunc():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


#Function to prepare PyTorch DataLoader for a Yelp Dataset
def PreparePytorchLd(mode, batch_size, loadeddata, classlabels,idfeature, jobs=0):
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    dataset = YelpDataset(loadeddata, classlabels, mode=mode, idfeature=idfeature)
    dataloader = DataLoader(dataset, batch_size,shuffle=shuffle, drop_last=False,
        num_workers=jobs, pin_memory=True)                            
    return dataloader 

class DLNetwork(nn.Module):
    def __init__(self, input_dim):
            super(DLNetwork, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 100),
                nn.ReLU(),
                nn.Linear(100, 2),
                nn.ReLU(),
                nn.Linear(2, 1),
                nn.Sigmoid()
            )
            self.criterion = nn.BCELoss()

    def forward(self, x):
            x = self.layers(x)
            return x.squeeze(1)

    def funcCalculateLoss(self, pred, targetlabels):
            return self.criterion(pred, targetlabels) 

def train(trainingset, YelpSet, model, config, devicetype):
    noOfEpochs = config['noOfEpochs']  
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_entropy = 1000.
    loss_record = {'train': [], 'dev': []}      
    early_stop_cnt = 0
    epoch = 0
    while epoch < noOfEpochs:
        model.train()                          
        for x, y in trainingset:                     
            optimizer.zero_grad()               
            x, y = x.to(devicetype), y.to(devicetype)   
            pred = model(x)                     
            BCE_loss = model.funcCalculateLoss(pred, y)  
            BCE_loss.backward()                 
            optimizer.step()                    
            loss_record['train'].append(BCE_loss.detach().cpu().item())

        
        dev_entropy = dev(YelpSet, model, devicetype)
        if dev_entropy < min_entropy:
            
            min_entropy = dev_entropy
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_entropy))
            torch.save(model.state_dict(), config['location_path'])  
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1
        loss_record['dev'].append(dev_entropy)
        if early_stop_cnt > config['StopAt']:
            break

    print('Stopped training after {} epochs'.format(epoch))
    return min_entropy, loss_record

#This function computes the loss of the given dataset
def dev(Yelpset, model, devicetype):
    model.eval()                               
    lossvalue = 0
    with torch.no_grad():
        for i, j in Yelpset:                        
            i = i.to(devicetype)
            prediction = model(i)  
            j = j.to(devicetype)                               
            entloss = model.funcCalculateLoss(prediction, j)  
            lossvalue += entloss.detach().cpu().item() * i.size(0)
    lossvalue = lossvalue/ len(Yelpset)
    return lossvalue 


#This function makes predictions on a given test set using a trained model.
def testfunc(set1, modeltype, devicetype):
    modeltype.eval()                                
    listprediction = []
    for val in set1:                           
        val = val.to(devicetype)                       
        with torch.no_grad():                   
            prediction = modeltype(val)                    
            listprediction.append(prediction.detach().cpu().numpy())   
    return np.concatenate(listprediction, axis=0)

#This function is used to save the predicted values to a csv file
def StorePredictions(preds, file, delimiter=','):
    print('Saved Predictions to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp, delimiter=delimiter)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


#This function is used to run our Deep learning model 
#It generates the evaluation metrics and plots them  
def DeepLearningModel(loadeddata, label, idfeature, top2_features, n_epoches=2, batch_size=10, type=''):
    devicetype = DeviceTypefunc()  
    rootdir = os.getcwd()
    os.makedirs(os.path.join(rootdir, 'CODE', 'models'), exist_ok=True)  

    config = {
        'noOfEpochs': n_epoches,   'batch_size': batch_size,  'optimizer': 'Adam',  'optim_hparas': {   
        },
        'StopAt': 500,  
        'location_path': os.path.join(rootdir, 'CODE', 'models', 'model.pth')  
    }

    X_train, X_test, y_train, y_test = train_test_split(loadeddata, label, test_size=0.3, random_state=42)
    trainingset = PreparePytorchLd('train', config['batch_size'], X_train, y_train, idfeature)
    DevSet = PreparePytorchLd('dev', config['batch_size'], X_train, y_train, idfeature)
    testingset = PreparePytorchLd('test', config['batch_size'], X_test, y_test, idfeature)

    FirstFeat, SecondFeat = top2_features
    Feature1 = X_test[FirstFeat].to_numpy()
    Feature2 = X_test[SecondFeat].to_numpy()

    model = DLNetwork(trainingset.dataset.dim).to(devicetype)  
    model_loss, model_loss_record = train(trainingset, DevSet, model, config, devicetype)
   
    model.load_state_dict(torch.load(config['location_path']))
    #plot_learning_curve(model_loss_record, title='deep model', type=type)
    preds = np.around(testfunc(testingset, model, devicetype))

   
    cdict = {0: 'darkgreen', 1: 'red'}
    plt.figure(figsize=(12, 8)) 
    fig, ax = plt.subplots()
    for k in np.unique(preds):
        list2 = list(np.where(preds == k)[0])
        ax.scatter(Feature1[list2], Feature2[list2], c=cdict[k], label=k, s=100)
    ax.legend()
    plt.legend(loc="upper left")
    plt.xlabel(FirstFeat)
    plt.ylabel(SecondFeat)
    plt.tight_layout() 
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/Image_{type}.png')
    StorePredictions(preds, "" + 'pred.csv')
    calculatemetrics(y_test, preds, type)


