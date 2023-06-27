import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np 
import copy

from utils.data import Dataset
from modules.logisticregressor.data_transformer import DataTransformer

from utils.criterions import ClassBalancedLoss

class Client():
    def __init__(self, name, state_dict, df_train, df_test, Criterion, Optimizer, learning_rate, continuous_columns=(), discrete_columns=(), target_column=None, local_epochs=1, batch_size=100):
        self.name = name
        self.epochs = local_epochs
        self.state_dict = state_dict
        
        df_train, df_val = train_test_split(df_train, test_size=0.2)
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        if target_column == None:
            target_column = df_train.columns[-1]
        y_train = df_train[target_column]
        X_train = df_train.drop(columns=target_column)
        y_val = df_val[target_column]
        X_val = df_val.drop(columns=target_column)
        y_test = df_test[target_column]
        X_test = df_test.drop(columns=target_column)

        '''
        uniques = [' <=50K', ' >50K'] #y_train.unique()
        for i, unique in enumerate(uniques):
            y_train = y_train.replace(unique, i)
            y_val = y_val.replace(unique, i)
            y_test = y_test.replace(unique, i)
        '''
        
        #self.data_transformer = DataTransformer()
        #self.data_transformer.fit(X_train, continuous_columns, discrete_columns)
        #X_train = self.data_transformer.transform(X_train)
        #X_val = self.data_transformer.transform(X_val)
        #X_test = self.data_transformer.transform(X_test)
        
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()
        y_test = y_test.to_numpy()

        trainset = Dataset(X_train, y_train)
        valset = Dataset(X_val, y_val)
        testset = Dataset(X_test, y_test)
        self.train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)#, num_workers=1)
        self.val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)#, num_workers=1)
        self.test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)#, num_workers=1)

        if Criterion == ClassBalancedLoss:
            unique_classes, n_samples_per_classes = np.unique(y_train, return_counts=True)
            if unique_classes[0] == 1.0:
                print(unique_classes)
                print(n_samples_per_classes)
            self.criterion = Criterion(beta=0.99, n_samples_per_classes=n_samples_per_classes, loss_type='sigmoid')
        else:
            self.criterion = Criterion()
        self.Optimizer = Optimizer
        self.learning_rate = learning_rate

        self.log = {
            'training': {
                'loss': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': []
            },
            'validation': {
                'loss': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': []
            },
            'test': {
                'loss': [],
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': []
            }
        }

    def train(self, model, device):
        #print('state_dict id: %i' % id(self.state_dict))
        model.load_state_dict(self.state_dict)
        model.train()
        optimizer = self.Optimizer(model.parameters(), lr=self.learning_rate)
        losses = []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        for _ in range(self.epochs):
            for X_train, y_true in self.train_loader:
                X_train = X_train.to(device)
                y_true = y_true.to(device)
                optimizer.zero_grad()
                y_pred = torch.squeeze(model(X_train))
                loss = self.criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()        
                losses.append(loss.item())
                y_true = y_true.detach().cpu()
                y_pred = y_pred.argmax(dim=1).detach().cpu()
                accuracies.append(accuracy_score(y_true=y_true, y_pred=y_pred))
                precisions.append(precision_score(y_true=y_true, y_pred=y_pred, zero_division=0))
                recalls.append(recall_score(y_true=y_true, y_pred=y_pred, zero_division=0))
                f1s.append(f1_score(y_true=y_true, y_pred=y_pred, zero_division=0))
        self.state_dict = copy.deepcopy(model.state_dict())
        loss = sum(losses)/len(losses) 
        accuracy = sum(accuracies)/len(accuracies)
        precision = sum(precisions)/len(precisions)
        recall = sum(recalls)/len(recalls)
        f1 = sum(f1s)/len(f1s)
        self.log['training']['loss'].append(loss)
        self.log['training']['accuracy'].append(accuracy)
        self.log['training']['precision'].append(precision)
        self.log['training']['recall'].append(recall)
        self.log['training']['f1'].append(f1)
        return loss, accuracy
    
    def validate(self, model, device):
        model.load_state_dict(self.state_dict)
        model.eval()
        losses = []
        accuracies = [] 
        precisions = []
        recalls = []
        f1s = []
        for X_val, y_true in self.val_loader:
            X_val = X_val.to(device)
            y_true = y_true.to(device)
            y_pred = torch.squeeze(model(X_val))
            loss = self.criterion(y_pred, y_true)
            losses.append(loss.item())
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(dim=1).detach().cpu()
            accuracies.append(accuracy_score(y_true=y_true, y_pred=y_pred))
            precisions.append(precision_score(y_true=y_true, y_pred=y_pred, zero_division=0))
            recalls.append(recall_score(y_true=y_true, y_pred=y_pred, zero_division=0))
            f1s.append(f1_score(y_true=y_true, y_pred=y_pred, zero_division=0))
        loss = sum(losses)/len(losses) 
        accuracy = sum(accuracies)/len(accuracies)
        precision = sum(precisions)/len(precisions)
        recall = sum(recalls)/len(recalls)
        f1 = sum(f1s)/len(f1s)
        self.log['validation']['loss'].append(loss)
        self.log['validation']['accuracy'].append(accuracy)
        self.log['validation']['precision'].append(precision)
        self.log['validation']['recall'].append(recall)
        self.log['validation']['f1'].append(f1)
        return loss, accuracy

    def test(self, model, device):
        model.load_state_dict(self.state_dict)
        model.eval()
        losses = []
        accuracies = [] 
        precisions = []
        recalls = []
        f1s = []
        for X_test, y_true in self.test_loader:
            X_test = X_test.to(device)
            y_true = y_true.to(device)
            y_pred = torch.squeeze(model(X_test), 1)
            loss = self.criterion(y_pred, y_true)
            losses.append(loss.item())
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(dim=1).detach().cpu()
            accuracies.append(accuracy_score(y_true=y_true, y_pred=y_pred))
            precisions.append(precision_score(y_true=y_true, y_pred=y_pred, zero_division=0))
            recalls.append(recall_score(y_true=y_true, y_pred=y_pred, zero_division=0))
            f1s.append(f1_score(y_true=y_true, y_pred=y_pred, zero_division=0))
        loss = sum(losses)/len(losses) 
        accuracy = sum(accuracies)/len(accuracies)
        precision = sum(precisions)/len(precisions)
        recall = sum(recalls)/len(recalls)
        f1 = sum(f1s)/len(f1s)
        self.log['test']['loss'].append(loss)
        self.log['test']['accuracy'].append(accuracy)
        self.log['test']['precision'].append(precision)
        self.log['test']['recall'].append(recall)
        self.log['test']['f1'].append(f1)
        return loss, accuracy
        

'''
from modules.logisticregressor.logisticregressor import LogisticRegressor
from torch.nn import BCELoss
from torch.optim import SGD

model = LogisticRegressor()
state_dict = model.state_dict()
df = pd.read_csv('data/adult/adult.csv')

client = Client(state_dict, df)

criterion = BCELoss()
optimizer = SGD(model.parameters(), lr=0.01)
        
client.train(model, criterion, optimizer)
'''
'''
df = pd.read_csv('data/adult/adult.csv')
df = df.drop(columns = ['fnlwgt', 'education'])
discrete_columns = ('workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income')
df_train, df_test = train_test_split(df, test_size=0.2)
Client(sate_dict=None, df_train=df, df_test=df, discrete_columns=discrete_columns, target_column='income')
'''