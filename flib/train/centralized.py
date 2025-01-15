import pandas as pd
from flib.utils import set_random_seed
from flib.train import Clients
import pandas as pd
import time
import pickle
import os
import yaml
from typing import Union, List

def centralized(seed:int, n_workers:int, client_type:str, client_names:Union[str, List[str]], client_params:Union[str, List[dict]]):
    set_random_seed(seed)
    Client = getattr(Clients, client_type)
    name = client_names[0] if isinstance(client_names, list) else client_names
    params = client_params[0] if isinstance(client_params, list) else client_params
    client = Client(name=name, seed=seed, **params)
    results = {client.name:  client.run(**params)}
    return results

if __name__ == '__main__':
    
    EXPERIMENT = '3_banks_homo_easy' # '30K_accts', '3_banks_homo_mid'
    CLIENT = 'LogRegClient'
    
    params = {
        'device': 'cuda:0',
        'nodes_train': f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/data/preprocessed/nodes_train.csv', 
        'nodes_test': f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/data/preprocessed/nodes_test.csv',
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
    
    t = time.time()
    results = centralized(seed=42, client=CLIENT, params=params)
    t = time.time() - t
    print('Done')
    print(f'Exec time: {t:.2f}s')
    results_dir = f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/results/centralized/{CLIENT}'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {results_dir}/results.pkl\n')
    