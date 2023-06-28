import pandas as pd
from utils.data import latent_dirchlet_allocation
from server import Server
from time import sleep
from modules.logisticregressor.logisticregressor import LogisticRegressor
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
import random 
import numpy as np
import os
from utils.data import latent_dirchlet_allocation
from utils.criterions import ClassBalancedLoss
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

def set_random_seed(seed:int=1):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: If you want every run to be exactly the same each time
        ##       uncomment the following lines
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_means_and_stds(log:dict, stages=['training', 'validation', 'test'], metrics=['loss', 'accuracy', 'precision', 'recall', 'f1']):
    means = {stage: {metric: np.array([]) for metric in metrics} for stage in stages}
    stds = {stage: {metric: np.array([]) for metric in metrics} for stage in stages}
    for stage in stages:
        for metric in metrics:
            n_rounds = len(log[next(iter(log))][stage][metric])
            for round in range(n_rounds):
                data = []
                for client in log.keys():
                    data.append(log[client][stage][metric][round])
                means[stage][metric] = np.append(means[stage][metric], np.mean(data))
                stds[stage][metric] = np.append(stds[stage][metric], np.std(data))
    return means, stds

def get_aggregated_confusion_matrices(log:dict, stages=['training', 'validation', 'test']):
    aggregated_confusion_matrices = {stage: [] for stage in stages}
    for stage in stages:
        n_rounds = len(log[next(iter(log))][stage]['confusion_matrix'])
        for round in range(n_rounds):
            aggregated_confusion_matrix = np.zeros((2, 2))
            for client in log.keys():
                confusion_matrix = log[client][stage]['confusion_matrix'][round]
                aggregated_confusion_matrix += confusion_matrix
            aggregated_confusion_matrix /= len(log.keys())
            aggregated_confusion_matrices[stage].append(aggregated_confusion_matrix.round())
    return aggregated_confusion_matrices
            
def main():
    
    # Spawning method 
    mp.set_start_method('spawn')
    
    # devices
    devices = [torch.device('cuda:0')]

    # seed
    seed = 42
    set_random_seed(seed)
    
    '''
    # Adult dataset
    # columns = [
    #   'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
    #   'marital-status', 'occupation', 'relationship', 'race', 'sex', 
    #   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    # ]
    df = pd.read_csv('data/adult/adult.csv')
    df = df.drop(columns = ['fnlwgt', 'education'])    
    df = df.replace(' ?', None)
    df = df.dropna()
    continuous_columns = ('age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week')
    discrete_columns = ('workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country')
    target_column = 'income'
    #df1 = df[df[target_column] == ' >50K']
    #df2 = df[df[target_column] == ' <=50K']
    #df2 = df2.sample(n=len(df1), random_state=42)
    #df = pd.concat([df1, df2], axis=0)
    df = pd.get_dummies(df, columns=list(discrete_columns))
    #df, _ = train_test_split(df, test_size=0.5)
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    '''

    # AML dataset
    DATASET = '10K_accts'
    df = pd.read_csv(f'/home/edvin/Desktop/flib/federated-learning/datasets/{DATASET}/data.csv')
    df1 = df[df['is_sar'] == 0.0]
    df2 = df[df['is_sar'] == 1.0]
    df = pd.concat([df1, df2.sample(n=295, random_state=seed)], axis=0)
    columns = df.columns
    df = df.drop(columns='account')
    #unique_banks = df['From Bank'].unique()
    #df = df[(df['From Bank'].isin(unique_banks[0:10])) & (df['To Bank'].isin(unique_banks[0:10]))]
    continuous_columns = tuple(columns[1:-1])
    #discrete_columns = ('From Bank', 'To Bank', 'Payment Format', 'Receiving Currency', 'Payment Currency')
    target_column = 'is_sar'
    #df1 = df[df[target_column] == 1]
    #df2 = df[df[target_column] == 0]
    #df2 = df2.sample(n=len(df1), random_state=42)
    #df = pd.concat([df1, df2], axis=0)
    #df = pd.get_dummies(df, columns=list(discrete_columns))
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)
    print('trainset imbalance:\n', df_train[target_column].value_counts())
    print('testset imbalance:\n', df_test[target_column].value_counts())
    
    # Hyperparameters centralized
    n_workers = 1 #1
    n_clients = 1 #5
    dfs_train = [df_train]
    n_rounds = 601
    eval_every = 10
    n_no_aggregation_rounds = n_rounds
    Model = LogisticRegressor
    Criterion = ClassBalancedLoss
    #loss_beta = 0.99
    Optimizer = SGD
    learning_rate = 0.1
    local_epochs = 1
    batch_size = 128
    
    set_random_seed(seed)
    
    server = Server(
        n_clients=n_clients, 
        n_rounds=n_rounds, 
        n_workers=n_workers, 
        Model=Model, 
        Criterion=Criterion,
        Optimizer=Optimizer,
        dfs_train=dfs_train, 
        df_test=df_test,
        continuous_columns=continuous_columns,
        discrete_columns=(),#discrete_columns,
        target_column=target_column,
        n_no_aggregation_rounds=n_no_aggregation_rounds,
        learning_rate = learning_rate,
        local_epochs = local_epochs,
        batch_size = batch_size,
        verbose = False,
        eval_every = eval_every,
        devices = devices
    )
    
    print('\ntraning centralized started\n')
    log1 = server.run()
    print('traning centralized done\n')
    
    # Hyperparameters isolated
    n_workers = 4
    n_clients = 12
    dfs_train = np.array_split(df_train, n_clients)


    set_random_seed(seed)
    
    server = Server(
        n_clients=n_clients, 
        n_rounds=n_rounds, 
        n_workers=n_workers, 
        Model=Model, 
        Criterion=Criterion,
        Optimizer=Optimizer,
        dfs_train=dfs_train, 
        df_test=df_test,
        continuous_columns=continuous_columns,
        discrete_columns=(),#discrete_columns,
        target_column=target_column,
        n_no_aggregation_rounds=n_no_aggregation_rounds,
        learning_rate = learning_rate,
        local_epochs = local_epochs,
        batch_size = batch_size,
        verbose = False,
        eval_every = eval_every,
        devices = devices
    )

    print('\ntraning isolated started\n')
    log2 = server.run()
    print('traning isolated done\n')
    
    # Hyperparameters federated
    n_no_aggregation_rounds = 0

    set_random_seed(seed)
    
    server = Server(
        n_clients=n_clients, 
        n_rounds=n_rounds, 
        n_workers=n_workers, 
        Model=Model, 
        Criterion=Criterion,
        Optimizer=Optimizer,
        dfs_train=dfs_train, 
        df_test=df_test,
        continuous_columns=continuous_columns,
        discrete_columns=(),#discrete_columns,
        target_column=target_column,
        n_no_aggregation_rounds=n_no_aggregation_rounds,
        learning_rate = learning_rate,
        local_epochs = local_epochs,
        batch_size = batch_size,
        verbose = False,
        eval_every = eval_every,
        devices = devices
    )
    
    print('\ntraning federated started\n')
    log3 = server.run()
    print('traning federated done\n')
    
    print('\ntraning isolated decision trees started\n')
    log4 = {}
    for i, df in enumerate(dfs_train):
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
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()
        y_test = y_test.to_numpy()

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_train)
        accuracy = accuracy_score(y_true=y_train, y_pred=y_pred)
        precision = precision_score(y_true=y_train, y_pred=y_pred, zero_division=0)
        recall = recall_score(y_true=y_train, y_pred=y_pred, zero_division=0)
        f1 = f1_score(y_true=y_train, y_pred=y_pred, zero_division=0)
        confusion = confusion_matrix(y_true=y_train, y_pred=y_pred)
        log4['client_%i' % i] = {'training': {'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'confusion_matrix': [confusion]}}
        
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_true=y_val, y_pred=y_pred)
        precision = precision_score(y_true=y_val, y_pred=y_pred, zero_division=0)
        recall = recall_score(y_true=y_val, y_pred=y_pred, zero_division=0)
        f1 = f1_score(y_true=y_val, y_pred=y_pred, zero_division=0)
        confusion = confusion_matrix(y_true=y_val, y_pred=y_pred)
        log4['client_%i' % i]['validation'] = {'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'confusion_matrix': [confusion]}

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(y_true=y_test, y_pred=y_pred, zero_division=0)
        recall = recall_score(y_true=y_test, y_pred=y_pred, zero_division=0)
        f1 = f1_score(y_true=y_test, y_pred=y_pred, zero_division=0)
        confusion = confusion_matrix(y_true=y_test, y_pred=y_pred)
        log4['client_%i' % i]['test'] = {'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'confusion_matrix': [confusion]}
    print('traning isolated decision trees done\n')

    rounds = [round for round in range(0, n_rounds, eval_every)]
    logs = [log1, log2, log3]
    colors = ['C0', 'C1', 'C2']
    labels = ['centralized', 'isolated', 'federated']
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for log, color, label in zip(logs, colors, labels):
        means, stds = get_means_and_stds(log, metrics=['accuracy', 'precision', 'recall', 'f1'])
        axs[0, 0].plot(rounds, means['test']['accuracy'], color=color, label=label)
        axs[0, 0].fill_between(rounds, means['test']['accuracy'] - stds['test']['accuracy'], means['test']['accuracy'] + stds['test']['accuracy'], color=color, alpha=0.2)
        axs[0, 1].plot(rounds, means['test']['precision'], color=color, label=label)
        axs[0, 1].fill_between(rounds, means['test']['precision'] - stds['test']['precision'], means['test']['precision'] + stds['test']['precision'], color=color, alpha=0.2)
        axs[1, 0].plot(rounds, means['test']['recall'], color=color, label=label)
        axs[1, 0].fill_between(rounds, means['test']['recall'] - stds['test']['recall'], means['test']['recall'] + stds['test']['recall'], color=color, alpha=0.2)
        axs[1, 1].plot(rounds, means['test']['f1'], color=color, label=label)
        axs[1, 1].fill_between(rounds, means['test']['f1'] - stds['test']['f1'], means['test']['f1'] + stds['test']['f1'], color=color, alpha=0.2)
    
    means, stds = get_means_and_stds(log4, metrics=['accuracy', 'precision', 'recall', 'f1'])
    x = np.array([rounds[0], rounds[-1]])
    y = np.array([means['test']['accuracy'][0], means['test']['accuracy'][0]])
    d = np.array([stds['test']['accuracy'][0], stds['test']['accuracy'][0]])
    axs[0, 0].plot(x, y, color='black', label='baseline')
    axs[0, 0].fill_between(x, y - d, y + d, color='black', alpha=0.2)
    y = np.array([means['test']['precision'][0], means['test']['precision'][0]])
    d = np.array([stds['test']['precision'][0], stds['test']['precision'][0]])
    axs[0, 1].plot(x, y, color='black', label='baseline')
    axs[0, 1].fill_between(x, y - d, y + d, color='black', alpha=0.2)
    y = np.array([means['test']['recall'][0], means['test']['recall'][0]])
    d = np.array([stds['test']['recall'][0], stds['test']['recall'][0]])
    axs[1, 0].plot(x, y, color='black', label='baseline')
    axs[1, 0].fill_between(x, y - d, y + d, color='black', alpha=0.2)
    y = np.array([means['test']['f1'][0], means['test']['f1'][0]])
    d = np.array([stds['test']['f1'][0], stds['test']['f1'][0]])
    axs[1, 1].plot(x, y, color='black', label='baseline')
    axs[1, 1].fill_between(x, y - d, y + d, color='black', alpha=0.2)

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for ax, metric in zip(axs.flat, metrics):
        ax.set_xlabel('round', fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.legend()
        ax.grid()
    
    plt.savefig('metrics.png')

    fig, axs = plt.subplots(1, 4, figsize=(16, 5))
    confusion_matrices = get_aggregated_confusion_matrices(log1)
    sns.heatmap(confusion_matrices['test'][-1], annot=True, annot_kws={"size": 16}, ax=axs[0], fmt='g', cmap='Blues', cbar=False)
    axs[0].set_title('centralized', fontsize=18)
    confusion_matrices = get_aggregated_confusion_matrices(log2)
    sns.heatmap(confusion_matrices['test'][-1], annot=True, annot_kws={"size": 16}, ax=axs[1], fmt='g', cmap='Oranges', cbar=False)
    axs[1].set_title('isolated', fontsize=18)
    confusion_matrices = get_aggregated_confusion_matrices(log3)
    sns.heatmap(confusion_matrices['test'][-1], annot=True, annot_kws={"size": 16}, ax=axs[2], fmt='g', cmap='Greens', cbar=False)
    axs[2].set_title('federated', fontsize=18)
    confusion_matrices = get_aggregated_confusion_matrices(log4)
    sns.heatmap(confusion_matrices['test'][-1], annot=True, annot_kws={"size": 16}, ax=axs[3], fmt='g', cmap='Greys', cbar=False)
    axs[3].set_title('baseline', fontsize=18)
    for ax in axs:
        ax.set_xlabel('prediction', fontsize=16)
        ax.set_ylabel('label', fontsize=16)
    fig.tight_layout(pad=2.0)
    plt.savefig('confusion_matrices.png')

    return

if __name__ == '__main__':
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    main()