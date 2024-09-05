import os
import sys
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
#from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Classifier:
    def __init__(self, dataset:pd.DataFrame):
        # data
        self.trainset, self.testset = dataset[0], dataset[1]
        # TODO: consider oversample
        self.X_train = self.trainset.drop(columns=['account', 'bank', 'is_sar']).to_numpy()
        self.y_train = self.trainset['is_sar'].to_numpy()
        self.X_test = self.testset.drop(columns=['account', 'bank', 'is_sar']).to_numpy()
        self.y_test = self.testset['is_sar'].to_numpy()
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        # model
        self.model = None


    def train(self, model='RandomForestClassifier', tune_hyperparameters=False):
        if model == 'RandomForestClassifier':
            model = getattr(sklearn.ensemble, model)
            if tune_hyperparameters:
                param_grid = {
                    'n_estimators': [5, 10], # 10, 100 
                    'criterion': ['gini', 'entropy'], # 'gini', 'entropy'
                    'max_depth': [5, 10], # None, 10, 20, 30
                    'min_samples_split': [2, 5], # 5, 10
                    'min_samples_leaf': [1, 5], # 5, 10 
                    'min_weight_fraction_leaf': [0.0, 0.1], # 0.1, 0.2
                    'max_features': ['sqrt', 'log2'], # 'log2', 'None'
                    'max_leaf_nodes': [None, 10], # 10, 100
                    'min_impurity_decrease': [0.0, 0.1], # 0.1, 0.2
                    'class_weight': ['balanced'], 
                    'random_state': [42],
                }
                grid = GridSearchCV(model(), param_grid, scoring='balanced_accuracy', verbose=1, n_jobs=-1)
                grid.fit(self.X_train, self.y_train)
                self.model = grid.best_estimator_
            else:
                self.model = model().fit(self.X_train, self.y_train)
        elif model == 'DecisionTreeClassifier':
            model = getattr(sklearn.tree, model)
            if tune_hyperparameters:
                param_grid = {
                    'criterion': ['gini', 'entropy'], # 'gini', 'entropy', 'log_loss'
                    'splitter': ['best', 'random'], # 'best', 'random'
                    'max_depth': [None, 10], # None, 10, 20, 30
                    'min_samples_split': [2, 5], # 5, 10
                    'min_samples_leaf': [1, 5], # 5, 10 
                    'min_weight_fraction_leaf': [0.0, 0.1], # 0.1, 0.2
                    'max_features': [None, 'sqrt', 'log2'], # 'log2', 'None'
                    'max_leaf_nodes': [None, 10], # 10, 100
                    'min_impurity_decrease': [0.0, 0.1], # 0.1, 0.2
                    'class_weight': ['balanced'], 
                    'random_state': [42],
                }
                grid = GridSearchCV(model(), param_grid, scoring='balanced_accuracy', verbose=1, n_jobs=-1)
                grid.fit(self.X_train, self.y_train)
                self.model = grid.best_estimator_
            else:
                self.model = model().fit(self.X_train, self.y_train)
        elif model == 'GradientBoostingClassifier':
            model = getattr(sklearn.ensemble, model)
            if tune_hyperparameters:
                param_grid = {
                    'loss': ['log_loss', 'exponential'], # 'log_loss', 'exponential'
                    'learning_rate': [0.01, 0.1], # [0.0, inf)
                    'n_estimators': [100, 200], # [1, inf)
                    'criterion': ['friedman_mse', 'squared_error'], # 'friedman_mse', 'squared_error'
                    'min_samples_split': [2, 5], # [2, inf)
                    'min_samples_leaf': [1, 5], # [1, inf)
                    'min_weight_fraction_leaf': [0.0, 0.1], # [0.0, 0.5]
                    'max_depth': [None, 3, 5], # None or [1, inf), tune for best performance
                    'min_impurity_decrease': [0.0, 0.1], # [0.0, inf)
                    'max_leaf_nodes': [None, 10], # None or [2, inf)
                    'random_state': [42],
                }
                grid = GridSearchCV(model(), param_grid, scoring='balanced_accuracy', verbose=1, n_jobs=-1)
                grid.fit(self.X_train, self.y_train)
                self.model = grid.best_estimator_
            else:
                self.model = model().fit(self.X_train, self.y_train)
        else:
            self.model = model.fit(self.X_train, self.y_train)
        return self.model


    def evaluate(self, operating_recall:int=0.8):
        y_pred = self.model.predict_proba(self.X_test)[:,1]
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred)
        if len(thresholds) == 1: # if only one threshold, all predict_proba are the same -> fpr = 1.0
            return 1.0, self.model.feature_importances_
        threshold = thresholds[np.argmax(recall <= operating_recall)]
        y_pred = (y_pred > threshold).astype(int)
        
        # calc recall
        recall = recall_score(self.y_test, y_pred)
        print(f'Recall: {recall:.4f}')
        
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        if tp+fp == 0:
            fpr = 1.0
        else:
            fpr = fp/(fp+tp)
        print(f'False positive rate: {fpr:.4f}')
        
        # Print the important features
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")
        for f in range(self.X_train.shape[1]):
            print(f'  {f}. {self.trainset.columns[indices[f]+2]} ({importances[indices[f]]})')
        
        avg_importance = importances.mean()
        avg_importance_error = abs(importances - avg_importance)
        sum_avg_importance_error = avg_importance_error.sum()
        print(f'Average importance error: {sum_avg_importance_error:.4f}')
        
        return fpr, importances
        