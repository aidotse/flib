from typing import Any, Dict

def centralized(seed: int, Client: Any, Model: Any, **kwargs) -> Dict[str, Dict]:
    
    client = Client(id='cen', seed=seed, Model=Model, **kwargs)
    results = {client.id:  client.run(**kwargs)}
    
    return results

if __name__ == '__main__':
    
    import argparse
    import os
    import pickle
    import time
    import yaml
    from flib import clients, models
    
    EXPERIMENT = '3_banks_homo_mid'
    CLIENT_TYPE = 'TorchGeometricClient' # 'TorchClient', 'TorchGeometricClient'
    MODEL_TYPE = 'GraphSAGE' # 'LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.', default=f'experiments/{EXPERIMENT}/config.yaml')
    parser.add_argument('--results', type=str, help='Path to results file.', default=f'experiments/{EXPERIMENT}/results/centralized/{MODEL_TYPE}/results.pkl')
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--client_type', type=str, help='Client class.', default=CLIENT_TYPE)
    parser.add_argument('--model_type', type=str, help='Model class.', default=MODEL_TYPE)
    args = parser.parse_args()
    
    t = time.time()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    kwargs = config[args.model_type]['default'] | config[args.model_type]['centralized']
    Client = getattr(clients, args.client_type)
    Model = getattr(models, args.model_type)
    results = centralized(seed=args.seed, Client=Client, Model=Model, **kwargs)
    
    t = time.time() - t
    
    print('Done')
    print(f'Exec time: {t:.2f}s')
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {args.results}\n')
