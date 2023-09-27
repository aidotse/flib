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
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import time
import argparse

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

def train_tree(dfs_train_decentralized, df_test_decentralized, target_column):
    log = {}
    for i, df in enumerate(dfs_train_decentralized):
        df_train, df_val = train_test_split(df, test_size=0.2)
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        if target_column == None:
            target_column = df_train.columns[-1]
        y_train = df_train[target_column]
        X_train = df_train.drop(columns=target_column)
        y_val = df_val[target_column]
        X_val = df_val.drop(columns=target_column)
        y_test = df_test_decentralized[target_column]
        X_test = df_test_decentralized.drop(columns=target_column)
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
        log['client_%i' % i] = {'training': {'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'confusion_matrix': [confusion]}}
        
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_true=y_val, y_pred=y_pred)
        precision = precision_score(y_true=y_val, y_pred=y_pred, zero_division=0)
        recall = recall_score(y_true=y_val, y_pred=y_pred, zero_division=0)
        f1 = f1_score(y_true=y_val, y_pred=y_pred, zero_division=0)
        confusion = confusion_matrix(y_true=y_val, y_pred=y_pred)
        log['client_%i' % i]['validation'] = {'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'confusion_matrix': [confusion]}

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(y_true=y_test, y_pred=y_pred, zero_division=0)
        recall = recall_score(y_true=y_test, y_pred=y_pred, zero_division=0)
        f1 = f1_score(y_true=y_test, y_pred=y_pred, zero_division=0)
        confusion = confusion_matrix(y_true=y_test, y_pred=y_pred)
        log['client_%i' % i]['test'] = {'accuracy': [accuracy], 'precision': [precision], 'recall': [recall], 'f1': [f1], 'confusion_matrix': [confusion]}
    return log

def plot_logs(logs, log4, labels, n_rounds=201, eval_every=20):
    rounds = [round for round in range(0, n_rounds, eval_every)]
    colors = ['C0', 'C1', 'C2']
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
    confusion_matrices = get_aggregated_confusion_matrices(logs[0])
    sns.heatmap(confusion_matrices['test'][-1], annot=True, annot_kws={"size": 16}, ax=axs[0], fmt='g', cmap='Blues', cbar=False)
    axs[0].set_title('centralized', fontsize=18)
    confusion_matrices = get_aggregated_confusion_matrices(logs[1])
    sns.heatmap(confusion_matrices['test'][-1], annot=True, annot_kws={"size": 16}, ax=axs[1], fmt='g', cmap='Oranges', cbar=False)
    axs[1].set_title('isolated', fontsize=18)
    confusion_matrices = get_aggregated_confusion_matrices(logs[2])
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

def run_experiment(args):
    curr_proc=mp.current_process()
    curr_proc.daemon=False

    set_random_seed(seed=args.seed)
    
    criterion_params = {
        'beta': args.beta
    }
    optimizer_params = {
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'dampening': args.dampening
    }
    
    server = Server(
        n_clients=args.n_clients, 
        n_rounds=args.n_rounds, 
        n_workers=args.n_workers, 
        Model=args.Model, 
        Criterion=args.Criterion,
        Optimizer=args.Optimizer,
        dfs_train=args.trainsets, 
        df_test=args.testsets,
        continuous_columns=args.continuous_columns,
        discrete_columns=args.discrete_columns,
        target_column=args.target_column,
        n_no_aggregation_rounds=args.n_no_aggregation_rounds,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        criterion_params=criterion_params,
        optimizer_params=optimizer_params,
        verbose=args.verbose,
        eval_every=args.eval_every,
        devices=args.devices,
        seed=args.seed
    )

    log = server.run()
    means, _ = get_means_and_stds(log=log, stages=['test'], metrics=['loss', 'accuracy', 'precision', 'recall', 'f1'])
    loss = means['test']['loss'][-1]
    accuracy = means['test']['accuracy'][-1]
    precision = means['test']['precision'][-1]
    recall = means['test']['recall'][-1]
    f1 = means['test']['f1'][-1]
    with open('param_sweep_cen.csv', 'a') as f:
        f.write('%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n' % (args.beta, args.momentum, args.weight_decay, args.dampening, args.learning_rate, loss, accuracy, precision, recall, f1))
    
    return

def main():
    
    # Spawning method 
    mp.set_start_method('spawn')
    
    # devices
    devices = [torch.device('cuda:0')]

    # seed
    seed = 42
    set_random_seed(seed=seed)

    # AML dataset
    DATASET = '10K_accts_super_easy_prepro2'
    #continuous_columns=[
    #    'num_outgoing_txs_1', 'num_outgoing_txs_2', 'num_outgoing_txs_3',
    #    'sum_outgoing_txs_1', 'sum_outgoing_txs_2', 'sum_outgoing_txs_3',
    #    'freq_outgoing_txs_1', 'freq_outgoing_txs_2', 'freq_outgoing_txs_3',
    #    'num_txs_1', 'num_txs_2', 'num_txs_3',
    #    'sum_txs_1', 'sum_txs_2', 'sum_txs_3',
    #    'freq_txs_1', 'freq_txs_2', 'freq_txs_3',
    #    'num_unique_counterparties_1', 'num_unique_counterparties_2', 'num_unique_counterparties_3',
    #    'freq_unique_counterparties_1', 'freq_unique_counterparties_2', 'freq_unique_counterparties_3',
    #    'num_phone_changes_1', 'num_phone_changes_2', 'num_phone_changes_3',
    #    'freq_phone_changes_1', 'freq_phone_changes_2', 'freq_phone_changes_3',
    #    'num_bank_changes_1', 'num_bank_changes_2', 'num_bank_changes_3',
    #    'freq_bank_changes_1', 'freq_bank_changes_2', 'freq_bank_changes_3',
    #]
    target_column = 'is_sar'
    banks = [
        'swedbank',
        'handelsbanken',
        'seb',
        'nordea',
        'danske',
        'länsförsäkringar',
        'ica',
        'sparbanken',
        'ålandsbanken',
        'marginalen',
        'svea',
        'skandia',
    ]

    trainsets = []
    testsets = []
    for bank in banks:
        dataset = pd.read_csv(f'/home/edvin/Desktop/flib/federated-learning/datasets/{DATASET}/{bank}.csv')
        continuous_columns = dataset.columns[1:-1]
        trainset, testset = train_test_split(dataset, test_size=0.2, random_state=seed)
        trainset = trainset.drop(columns='account')
        testset = testset.drop(columns='account')
        trainset = trainset.sample(frac=1, random_state=seed).reset_index(drop=True)
        testset = testset.sample(frac=1, random_state=seed).reset_index(drop=True)
        trainsets.append(trainset)
        testsets.append(testset)
    testset = pd.concat(testsets, axis=0)
    trainset = pd.concat(trainsets, axis=0)
    
    #param_grid = ParameterGrid({
    #    'devices': [devices],
    #    'n_workers': [1],
    #    'n_clients': [1],
    #    'trainsets': [[trainset]],
    #    'testsets': [testset],
    #    'continuous_columns': [continuous_columns],
    #    'discrete_columns': [()],
    #    'target_column': [target_column],
    #    'type': ['federated'],
    #    'n_rounds': [101],
    #    'eval_every': [10],
    #    'n_no_aggregation_rounds': [101],
    #    'Model': [LogisticRegressor],
    #    'Criterion': [ClassBalancedLoss],
    #    'Optimizer': [SGD],
    #    'local_epochs': [1],
    #    'beta': [0.9, 0.933333, 0.966666, 0.999999],
    #    'momentum': [0.1, 0.3, 0.6, 0.9],
    #    'weight_decay': [0.0001, 0.001, 0.01, 0.1],
    #    'dampening': [0.0001, 0.001, 0.01, 0.1],
    #    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    #    'batch_size': [2048],
    #    'seed': [seed],
    #    'verbose': [False],
    #})
    #
    #args_list = []
    #df = pd.read_csv('param_sweep_cen.csv')
    #for i, params in enumerate(param_grid):
    #    if df[(df['beta']==params['beta']) & (df['momentum']==params['momentum']) & (df['weight_decay']==params['weight_decay']) & (df['dampening']==params['dampening']) & (df['learning_rate']==params['learning_rate'])].empty:
    #        parser = argparse.ArgumentParser()
    #        for key, value in params.items():
    #            parser.add_argument(f'--{key}', type=type(value), default=value, help=f'enter {key}')
    #        args = parser.parse_args()
    #        args_list.append(args)
    #
    #print('number of experiments:', len(args_list))
    #
    #with open('param_sweep_cen.csv', 'a') as f:
    #    f.write('beta,momentum,weight_decay,dampening,learning_rate,loss,accuracy,precision,recall,f1\n')
    #
    #start_time = time.time()
    #with mp.Pool(20) as p:
    #    p.map(run_experiment, args_list)
    #end_time = time.time()
    #print(f'elapsed time: {end_time - start_time}')
    #print()

    N_ROUNDS = 101
    EVAL_EVERY = 20

    print('traning centralized')
    set_random_seed(seed)
    criterion_params = {
        'beta': 0.5
    }
    optimizer_params = {
        'momentum': 0.6,
        'weight_decay': 0.1,
        'dampening': 0.1
    }
    server = Server(
        devices=devices,
        n_workers=1,
        n_clients=1,
        dfs_train=[trainset],
        df_test=testset,
        continuous_columns=continuous_columns,
        discrete_columns=(),
        target_column=target_column,
        n_rounds=N_ROUNDS,
        eval_every=EVAL_EVERY,
        n_no_aggregation_rounds=N_ROUNDS,
        Model=LogisticRegressor,
        Criterion=ClassBalancedLoss,
        criterion_params=criterion_params,
        Optimizer=SGD,
        optimizer_params=optimizer_params,
        local_epochs=1,
        learning_rate=10.0,
        batch_size=2048,
        seed=seed,
        verbose=False,
    )
    log_cen = server.run()
    print()
    
    print('traning isolated')
    set_random_seed(seed)
    criterion_params = {
        'beta': 0.5
    }
    optimizer_params = {
        'momentum': 0.6,
        'weight_decay': 0.1,
        'dampening': 0.1,
    }
    server = Server(
        devices=devices,
        n_workers=6,
        n_clients=12,
        dfs_train=trainsets,
        df_test=testset,
        continuous_columns=continuous_columns,
        discrete_columns=(),
        target_column=target_column,
        n_rounds=N_ROUNDS,
        eval_every=EVAL_EVERY,
        n_no_aggregation_rounds=N_ROUNDS,
        Model=LogisticRegressor,
        Criterion=ClassBalancedLoss,
        criterion_params=criterion_params,
        Optimizer=SGD,
        optimizer_params=optimizer_params,
        local_epochs=1,
        learning_rate=10.0,
        batch_size=2048,
        seed=seed,
        verbose=False,
    )
    log_iso = server.run()
    print()
    
    print('traning federated')
    set_random_seed(seed)
    criterion_params = {
        'beta': 0.5
    }
    optimizer_params = {
        'momentum': 0.6,
        'weight_decay': 0.1,
        'dampening': 0.1,
    }
    server = Server(
        devices=devices,
        n_workers=6,
        n_clients=12,
        dfs_train=trainsets,
        df_test=testset,
        continuous_columns=continuous_columns,
        discrete_columns=(),
        target_column=target_column,
        n_rounds=N_ROUNDS,
        eval_every=EVAL_EVERY,
        n_no_aggregation_rounds=0,
        Model=LogisticRegressor,
        Criterion=ClassBalancedLoss,
        criterion_params=criterion_params,
        Optimizer=SGD,
        optimizer_params=optimizer_params,
        local_epochs=1,
        learning_rate=10.0,
        batch_size=2048,
        seed=seed,
        verbose=False,
    )
    #start_time = time.time()
    log_fed = server.run()
    #end_time = time.time()
    print()
    #print('elapsed time:', end_time - start_time)
    #print(log_fed['client_0']['test']['accuracy'][-1])

    print('traning tree')
    set_random_seed(seed)
    log4 = train_tree(dfs_train_decentralized=trainsets, df_test_decentralized=testset, target_column=target_column)
    print(' done')
    
    logs = [log_cen, log_iso, log_fed]
    labels = ['centralized', 'isolation', 'federated']
    plot_logs(logs, log4, labels, n_rounds=N_ROUNDS, eval_every=EVAL_EVERY)
    
    return

if __name__ == '__main__':
    #os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    main()