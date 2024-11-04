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

def federated(train_dfs, val_dfs=[], test_dfs=[], seed=42, n_workers=3, n_rounds=100, eval_every=10, client='LogRegClient', **kwargs):
    
    set_random_seed(seed)
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print(f'Start method already set to {mp.get_start_method()}')
        pass
    
    Client = getattr(Clients, client)
    
    # init clients
    clients = []
    for i, train_df in enumerate(train_dfs):
        val_df = val_dfs[i] if i < len(val_dfs) else None
        test_df = test_dfs[i] if i < len(test_dfs) else None
        client = Client(
            name=f'c{i}',
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            **kwargs
        )
        clients.append(client)
    
    # init server
    server = Server(
        clients=clients,
        n_workers=n_workers
    )
    
    # run
    results = server.run(n_rounds=n_rounds, eval_every=eval_every, **kwargs)
    
    return results

if __name__ == '__main__':
    
    DATASET = '3_banks_homo_mid' # '30K_accts', '3_banks_homo_mid'
    
    # for debugging
    c0_train_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_nodes_train.csv'
    c0_train_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_edges_train.csv'
    c1_train_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_nodes_train.csv'
    c1_train_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_edges_train.csv'
    c2_train_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_nodes_train.csv'
    c2_train_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_edges_train.csv'
    c0_test_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_nodes_test.csv'
    c0_test_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_edges_test.csv'
    c1_test_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_nodes_test.csv'
    c1_test_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_edges_test.csv'
    c2_test_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_nodes_test.csv'
    c2_test_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_edges_test.csv'
    
    train_dfs = [
        (pd.read_csv(c0_train_nodes_csv).drop(columns=['account', 'bank']), pd.read_csv(c0_train_edges_csv)),
        (pd.read_csv(c1_train_nodes_csv).drop(columns=['account', 'bank']), pd.read_csv(c1_train_edges_csv)),
        (pd.read_csv(c2_train_nodes_csv).drop(columns=['account', 'bank']), pd.read_csv(c2_train_edges_csv)),
    ]
    test_dfs = [
        (pd.read_csv(c0_test_nodes_csv).drop(columns=['account', 'bank']), pd.read_csv(c0_test_edges_csv)),
        (pd.read_csv(c1_test_nodes_csv).drop(columns=['account', 'bank']), pd.read_csv(c1_test_edges_csv)),
        (pd.read_csv(c2_test_nodes_csv).drop(columns=['account', 'bank']), pd.read_csv(c2_test_edges_csv)),
    ]
    
    params = {
        'lr_patience': 100,
        'es_patience': 100,
        'device': 'cuda:0',
        'batch_size': 128,
        'hidden_dim': 64,
        'optimizer': 'Adam',
        'optimizer_params': {
            'lr': 0.01,
            'weight_decay': 0.0,
            'amsgrad': False,
        },
        'criterion': 'CrossEntropyLoss',
        'criterion_params': {}
    }
    
    t = time.time()
    results = federated(train_dfs=train_dfs, test_dfs=test_dfs, seed=42, n_workers=1, n_rounds=100, eval_every=10, client='GraphSAGEClient', **params)
    t = time.time() - t
    print('Done')
    print(f'Exec time: {t:.2f}s')
    results_dir = f'/home/edvin/Desktop/flib/experiments/results/{DATASET}/federated/GraphSAGEClient'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {results_dir}/results.pkl\n')