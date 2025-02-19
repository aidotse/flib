from typing import Any, Dict

def federated(seed:int, Server: Any, Client: Any, Model: Any, n_workers: int = None, **kwargs) -> Dict:
    
    client_configs = kwargs.pop('clients')
    clients = []
    for id, config in client_configs.items():
        params = kwargs | config
        clients.append(Client(id=id, seed=seed, Model=Model, **params))
    
    if n_workers is None:
        n_workers = len(clients)

    server = Server(seed=seed, Model=Model, clients=clients, n_workers=n_workers, **kwargs)
    results = server.run(**kwargs)
    
    return results

if __name__ == '__main__':
    
    import argparse
    import multiprocessing as mp
    import os
    import pickle
    import time
    import yaml
    from flib import servers, clients, models
    
    mp.set_start_method('spawn', force=True)
    
    EXPERIMENT = '3_banks_homo_mid'
    SERVER_TYPE = 'TorchServer'
    CLIENT_TYPE = 'TorchGeometricClient' # 'TorchClient', 'TorchGeometricClient'
    MODEL_TYPE = 'GCN' # 'LogisticRegressor', 'MLP', 'GCN'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.', default=f'experiments/{EXPERIMENT}/config.yaml')
    parser.add_argument('--results', type=str, help='Path to results file.', default=f'experiments/{EXPERIMENT}/results/federated/{MODEL_TYPE}/results.pkl')
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--n_workers', type=int, help='Number of workers. Default is number of clients.', default=None)
    parser.add_argument('--server_type', type=str, help='Server class.', default=SERVER_TYPE)
    parser.add_argument('--client_type', type=str, help='Client class.', default=CLIENT_TYPE)
    parser.add_argument('--model_type', type=str, help='Model class.', default=MODEL_TYPE)
    args = parser.parse_args()
    
    t = time.time()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    kwargs = config[args.model_type]['default'] | config[args.model_type]['federated']
    Server = getattr(servers, args.server_type)
    Client = getattr(clients, args.client_type)
    Model = getattr(models, args.model_type)
    results = federated(seed=args.seed, Server=Server, Client=Client, Model=Model, n_workers=args.n_workers, **kwargs)
    
    t = time.time() - t
    
    print('Done')
    print(f'Exec time: {t:.2f}s')
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {args.results}\n')
