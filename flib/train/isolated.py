from flib.train import Clients
from flib.utils import set_random_seed
import multiprocessing as mp
import numpy as np
import copy
import pandas as pd
import time
import pickle
import os
from typing import List

def train_clients(clients, kwargs):
    client_names = []
    results = []
    for client in clients:
        client_names.append(client.name)
        results.append(client.run(**kwargs))#['clients'][client.name]))
    return client_names, results

def isolated(seed:int, n_workers:int, client_type:str, client_names:List[str], client_params:List[dict]): #(train_dfs, val_dfs=[], test_dfs=[], seed=42, n_workers=3, client='LogRegClient', **kwargs):
    set_random_seed(seed)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    Client = getattr(Clients, client_type)
    clients = []
    for name, params in zip(client_names, client_params):
        client = Client(
            name=name,
            seed=seed,
            **params
        )
        clients.append(client)
    with mp.Pool(n_workers) as p:
        client_splits = np.array_split(clients, n_workers)
        results = p.starmap(train_clients, [(client_split, client_params[0]) for client_split in client_splits]) 
    results = {client_name: result for client_names, results in results for client_name, result in zip(client_names, results)}
    return results

if __name__ == '__main__':
    
    EXPERIMENT = '3_banks_homo_easy' # '30K_accts', '3_banks_homo_mid'
    CLIENT = 'LogRegClient'
    
    client_names = ['c0', 'c1', 'c2']
    client_params = [
        {
            'device': 'cuda:0',
            'nodes_train': f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/clients/c0/data/preprocessed/nodes_train.csv', 
            'nodes_test': f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/clients/c0/data/preprocessed/nodes_test.csv',
            'valset_size': 0.2,
            'batch_size': 2048,
            'optimizer': 'Adam',
            'optimizer_params': {
                'lr': 0.01,
                'weight_decay': 0.0,
                'amsgrad': False,
            },
            'criterion': 'CrossEntropyLoss',
            'criterion_params': {},
            'rounds': 100,
            'eval_every': 10,
            'lr_patience': 100,
            'es_patience': 100,
            'hidden_dim': 64,
        },
        {
            'device': 'cuda:0',
            'nodes_train': f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/clients/c1/data/preprocessed/nodes_train.csv', 
            'nodes_test': f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/clients/c1/data/preprocessed/nodes_test.csv',
            'valset_size': 0.2,
            'batch_size': 2048,
            'optimizer': 'Adam',
            'optimizer_params': {
                'lr': 0.01,
                'weight_decay': 0.0,
                'amsgrad': False,
            },
            'criterion': 'CrossEntropyLoss',
            'criterion_params': {},
            'rounds': 100,
            'eval_every': 10,
            'lr_patience': 100,
            'es_patience': 100,
            'hidden_dim': 64,
        },
        {
            'device': 'cuda:0',
            'nodes_train': f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/clients/c2/data/preprocessed/nodes_train.csv', 
            'nodes_test': f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/clients/c2/data/preprocessed/nodes_test.csv',
            'valset_size': 0.2,
            'batch_size': 2048,
            'optimizer': 'Adam',
            'optimizer_params': {
                'lr': 0.01,
                'weight_decay': 0.0,
                'amsgrad': False,
            },
            'criterion': 'CrossEntropyLoss',
            'criterion_params': {},
            'rounds': 100,
            'eval_every': 10,
            'lr_patience': 100,
            'es_patience': 100,
            'hidden_dim': 64,
        }
    ]
    
    t = time.time()
    results = isolated(seed=42, n_workers=3, client_type=CLIENT, client_names=client_names, client_params=client_params)
    t = time.time() - t
    print('Done')
    print(f'Exec time: {t:.2f}s')
    results_dir = f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/results/isolated/{CLIENT}'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {results_dir}/results.pkl\n')
    