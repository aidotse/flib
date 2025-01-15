from flib.train import Clients
from flib.train.servers import Server
from flib.utils import set_random_seed
import multiprocessing as mp
import torch
import os
import pickle
import optuna
import time
import pandas as pd
from typing import List

def federated(seed:int, n_workers:int, client_type:str, client_names:List[str], client_params:List[dict]):
    set_random_seed(seed)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print(f'Start method already set to {mp.get_start_method()}')
        pass
    Client = getattr(Clients, client_type)
    clients = []
    for name, params in zip(client_names, client_params):
        client = Client(name=name, seed=seed, **params)
        clients.append(client)
    server = Server(clients=clients, n_workers=n_workers)
    results = server.run(**client_params[0])
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
    results = federated(seed=42, n_workers=3, client_type=CLIENT, client_names=client_names, client_params=client_params)
    t = time.time() - t
    print('Done')
    print(f'Exec time: {t:.2f}s')
    results_dir = f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/results/federated/{CLIENT}'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {results_dir}/results.pkl\n')
    