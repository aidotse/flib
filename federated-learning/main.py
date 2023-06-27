import pandas as pd
from utils.data import latent_dirchlet_allocation
from server import Server
from time import sleep
from modules.logisticregressor.logisticregressor import LogisticRegressor
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
import random 
import numpy as np
import os
from utils.criterions import ClassBalancedLoss

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
    df = pd.concat([df1, df2.sample(n=295)], axis=0)
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
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print('trainset imbalance:\n', df_train[target_column].value_counts())
    print('testset imbalance:\n', df_test[target_column].value_counts())

    # Hyperparameters centralized
    n_workers = 1 #1
    n_clients = 1 #5
    n_rounds = 301
    eval_every = 10
    n_no_aggregation_rounds = n_rounds
    Model = LogisticRegressor
    Criterion = ClassBalancedLoss
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
        df_train=df_train, 
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

    set_random_seed(seed)
    
    server = Server(
        n_clients=n_clients, 
        n_rounds=n_rounds, 
        n_workers=n_workers, 
        Model=Model, 
        Criterion=Criterion,
        Optimizer=Optimizer,
        df_train=df_train, 
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
        df_train=df_train, 
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
    

    rounds = [round for round in range(0, n_rounds, eval_every)]
    logs = [log1, log2, log3]
    colors = ['C0', 'C1', 'C2']
    labels = ['centralized', 'isolated', 'federated']
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for log, color, label in zip(logs, colors, labels):
        means, stds = get_means_and_stds(log)
        axs[0, 0].plot(rounds, means['test']['accuracy'], color=color, label=label)
        axs[0, 0].fill_between(rounds, means['test']['accuracy'] - stds['test']['accuracy'], means['test']['accuracy'] + stds['test']['accuracy'], color=color, alpha=0.2)
        axs[0, 1].plot(rounds, means['test']['precision'], color=color, label=label)
        axs[0, 1].fill_between(rounds, means['test']['precision'] - stds['test']['precision'], means['test']['precision'] + stds['test']['precision'], color=color, alpha=0.2)
        axs[1, 0].plot(rounds, means['test']['recall'], color=color, label=label)
        axs[1, 0].fill_between(rounds, means['test']['recall'] - stds['test']['recall'], means['test']['recall'] + stds['test']['recall'], color=color, alpha=0.2)
        axs[1, 1].plot(rounds, means['test']['f1'], color=color, label=label)
        axs[1, 1].fill_between(rounds, means['test']['f1'] - stds['test']['f1'], means['test']['f1'] + stds['test']['f1'], color=color, alpha=0.2)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for ax, metric in zip(axs.flat, metrics):
        ax.set_xlabel('round')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid()
    plt.savefig('results.png')

    return

if __name__ == '__main__':
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    main()