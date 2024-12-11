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

from flib.train import HyperparamTuner, isolated

class Classifier:
    def __init__(self, dataset:pd.DataFrame, results_dir:str):
        # data
        self.trainset, self.testset = dataset[0].drop(columns=['account', 'bank']), dataset[1].drop(columns=['account', 'bank'])
        self.X_train = self.trainset.drop(columns=['is_sar']).to_numpy()
        self.y_train = self.trainset['is_sar'].to_numpy()
        self.X_test = self.testset.drop(columns=['is_sar']).to_numpy()
        self.y_test = self.testset['is_sar'].to_numpy()
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        # model
        self.model = None
        
        # results dir for hyperparameter tuning
        self.results_dir = results_dir


    def train(self, model='DecisionTreeClassifier', tune_hyperparameters=False, n_trials=10):
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
                self.model = model(class_weight='balanced', random_state=42).fit(self.X_train, self.y_train)
        elif model == 'DecisionTreeClassifier':
            
            if tune_hyperparameters:
                #param_grid = {
                #    'criterion': ['gini', 'entropy', 'log_loss'], # 'gini', 'entropy', 'log_loss'
                #    'splitter': ['best', 'random'], # 'best', 'random'
                #    'max_depth': [None, 6, 8, 10, 12, 14, 16], # None, 10, 20, 30
                #    'class_weight': ['balanced'], 
                #    'random_state': [42],
                #}
                #grid = GridSearchCV(model(), param_grid, scoring='balanced_accuracy', verbose=1, n_jobs=-1)
                #grid.fit(self.X_train, self.y_train)
                #self.model = grid.best_estimator_
                params = {
                    'default': {
                        'criterion': 'gini',
                        'splitter': 'best',
                        'max_depth': None,
                        'random_state': 42,
                        'class_weight': 'balanced'
                    },
                    'search_space': {
                        'criterion': ['gini', 'entropy', 'log_loss'],
                        'splitter': ['best', 'random'],
                        'max_depth': (1, 100),
                        #'class_weight': ['balanced', None]
                    }
                }
                hyperparams = self.tune_hyperparameters(model, params, n_trials)
                for param in params['default']:
                    if param not in hyperparams:
                        hyperparams[param] = params['default'][param]
            else:
                hyperparams = {
                    'criterion': 'gini',
                    'splitter': 'best',
                    'max_depth': None,
                    'random_state': 42,
                    'class_weight': 'balanced'
                },
            model = getattr(sklearn.tree, model)
            self.model = model(**hyperparams).fit(self.X_train, self.y_train)
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
                self.model = model(random_state=42).fit(self.X_train, self.y_train)
        else:
            self.model = model.fit(self.X_train, self.y_train)
        return self.model


    def evaluate(self, utility='fpr'):
        if utility == 'fpr':
            y_pred = self.model.predict_proba(self.X_test)[:,1]
            y_pred = (y_pred > 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            if tp+fp == 0:
                score = 1.0
            else:
                score = fp/(fp+tp)
        if utility == 'pr-auc':
            y_pred = self.model.predict_proba(self.X_test)[:,1]
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred)
            score = sklearn.metrics.auc(recall, precision)
        if utility == 'ap':
            y_pred = self.model.predict_proba(self.X_test)[:,1]
            score = sklearn.metrics.average_precision_score(self.y_test, y_pred)
        importances = self.model.feature_importances_
        return score, importances


    def tune_hyperparameters(self, model, params, n_trials):
        study_name = model
        storage = 'sqlite:///' + os.path.join(self.results_dir, model+'_hp_study.db')
        if os.path.exists(storage.replace('sqlite:///', '')):
            os.remove(storage.replace('sqlite:///', ''))
        val_df = self.trainset.sample(frac=0.2, random_state=42)
        train_df = self.trainset.drop(val_df.index)
        hyperparamtuner = HyperparamTuner(
            study_name = study_name,
            obj_fn = isolated,
            train_dfs = [train_df],
            val_dfs = [val_df],
            seed = 42,
            storage = storage,
            client = model.replace('Classifier', 'Client'),
            n_workers = 1,
            device = 'cuda:0',
            params = params
        )
        best_trials = hyperparamtuner.optimize(n_trials=n_trials)
        best_trials_file = os.path.join(self.results_dir, model+'_best_trials.txt')
        if os.path.exists(best_trials_file):
            os.remove(best_trials_file)
        with open(best_trials_file, 'w') as f:
            for trial in best_trials:
                print(f'\ntrial: {trial.number}')
                f.write(f'\ntrial: {trial.number}\n')
                print(f'values: {trial.values}')
                f.write(f'values: {trial.values}\n')
                for param in trial.params:
                    f.write(f'{param}: {trial.params[param]}\n')
                    print(f'{param}: {trial.params[param]}')
        print()
        os.remove(storage.replace('sqlite:///', ''))
        os.remove(best_trials_file)
        return best_trials[-1].params


def main(dataset, results_dir):
    classifier = Classifier(dataset, results_dir)
    classifier.train(model='DecisionTreeClassifier', tune_hyperparameters=True, n_trials=10)
    score, importances = classifier.evaluate(utility='ap')
    avg_importance = importances.mean()
    print(f'Utility score: {score}')
    print(f'Average feature importance: {avg_importance}')
    pass

if __name__ == '__main__':
    DATASET = '30K_accts'
    train_df = pd.read_csv(f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_nodes_train.csv')
    test_df = pd.read_csv(f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_nodes_test.csv')
    dataset = [train_df, test_df]
    results_dir = f'/home/edvin/Desktop/flib/experiments/param_files/{DATASET}'
    main(dataset, results_dir)