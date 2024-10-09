from flib.train import Clients
from flib.utils import set_random_seed
import multiprocessing as mp
import numpy as np
import copy

def train_clients(clients, kwargs):
    client_names = []
    results = []
    for client in clients:
        client_names.append(client.name)
        results.append(client.run(**kwargs))
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
            **kwargs
        )
        clients.append(client)
    
    with mp.Pool(n_workers) as p:
        
        client_splits = np.array_split(clients, n_workers)
        
        state_dict = clients[0].get_state_dict()
        for client in clients[1:]:
            client.load_state_dict(copy.deepcopy(state_dict))
        
        results = p.starmap(train_clients, [(client_split, kwargs) for client_split in client_splits])
    
    results = {client_name: result for client_names, results in results for client_name, result in zip(client_names, results)}
    
    return results