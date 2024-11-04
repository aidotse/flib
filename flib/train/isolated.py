from flib.train import Clients
from flib.utils import set_random_seed
import multiprocessing as mp
import numpy as np
import copy
import pandas as pd
import time
import pickle
import os

def train_clients(clients, kwargs):
    client_names = []
    results = []
    for client in clients:
        client_names.append(client.name)
        results.append(client.run(**kwargs['clients'][client.name]))
    return client_names, results

def isolated(train_dfs, val_dfs=[], test_dfs=[], seed=42, n_workers=3, client='LogRegClient', **kwargs):
    
    set_random_seed(seed)
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
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
            **kwargs['clients'][f'c{i}']
        )
        clients.append(client)
    
    with mp.Pool(n_workers) as p:
        
        client_splits = np.array_split(clients, n_workers)
        
        #state_dict = clients[0].get_state_dict()
        #for client in clients[1:]:
        #    client.load_state_dict(copy.deepcopy(state_dict))
        
        results = p.starmap(train_clients, [(client_split, kwargs) for client_split in client_splits])
    
    results = {client_name: result for client_names, results in results for client_name, result in zip(client_names, results)}
    
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
        'rounds': 100,
        'eval_every': 10,
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
    results = isolated(train_dfs=train_dfs, test_dfs=test_dfs, seed=42, n_workers=3, client='GraphSAGEClient', **params)
    t = time.time() - t
    print('Done')
    print(f'Exec time: {t:.2f}s')
    results_dir = f'/home/edvin/Desktop/flib/experiments/results/{DATASET}/isolated/GraphSAGEClient'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {results_dir}/results.pkl\n')