import os
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
#from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Classifier:
    def __init__(self, dataset:pd.DataFrame, model:str='RandomForestClassifier', operating_recall:int=0.8, target_fpr:int=0.95, ):
        self.trainset, self.testset = dataset[0], dataset[1]
        self.model = getattr(sklearn.ensemble, model)        


    def train(trainset, model=['RandomForest'], tune_hyperparameters=False):

        X = trainset.drop(columns=['account', 'bank', 'is_sar']).to_numpy()
        y = trainset['is_sar'].to_numpy()

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        #ros = RandomOverSampler(sampling_strategy=0.7)
        #X, y = ros.fit_resample(X, y)

        param_grid = {
            'n_estimators': [10, 100], 
            'criterion': ['gini', 'entropy'], # log_loss
            'max_depth': [10, 20], # 30
            'min_samples_split': [2, 5], # 10
            'min_samples_leaf': [5, 10], # 15 
            'min_weight_fraction_leaf': [0.0, 0.1], # 0.2
            'max_features': ['sqrt', None], # log2
            'max_leaf_nodes': [10, None], # 100
            'min_impurity_decrease': [0.0, 0.1], # 0.2
            'class_weight': ['balanced'], 
            'random_state': [42],
        }

        grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring='balanced_accuracy', verbose=1, n_jobs=-1)
        grid.fit(X, y)

        print(grid.best_params_)
        print(grid.best_score_)

        return grid.best_estimator_


    def evaluate(self, operating_recall:int=0.8, target_fpr:int=0.95):
        return
    
    
    def precision_after_recall(self, X, y_true):
        y_pred = self.model.predict_proba(X)[:,1]
        precision, recall, threshold = precision_recall_curve(y_true, y_pred)
        recall = 0.75
        idx = np.argmax(recall <= recall)
        return precision[idx]
        