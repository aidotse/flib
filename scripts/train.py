import argparse
import multiprocessing as mp
import os
import pickle
import time
import yaml
from flib import servers, clients, models
from flib.train import centralized, federated, isolated

def main():
    
    mp.set_start_method('spawn', force=True)
    
    EXPERIMENT = '12_banks_homo_mid'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.', default=f'experiments/{EXPERIMENT}/config.yaml')
    parser.add_argument('--model_types', nargs='+', help='Types of models to train.', default=['LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE']) # 'LogisticRegressor', 'MLP', 'GCN', 'GAT', 'GraphSAGE'
    parser.add_argument('--n_workers', type=int, help='Number of workers. Defaults to number of clients.', default=4)
    parser.add_argument('--results_dir', type=str, default=f'experiments/{EXPERIMENT}/results')
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "isolated", "centralized" or "federated".', default=['centralized', 'federated', 'isolated']) # 'centralized', 'federated', 'isolated'
    args = parser.parse_args()
    
    print('\nParsed arguments:')
    for arg, val in vars(args).items():
        print(f'{arg}: {val}')
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for model_type in args.model_types:
        if 'centralized' in args.settings:
            print(f'\nTraining {model_type} in centralized setting.')
            t = time.time()
            kwargs = config[model_type]['default'] | config[model_type]['centralized']
            results = centralized(
                seed = args.seed, 
                Client = getattr(clients, config[model_type]['default']['client_type']),
                Model = getattr(models, model_type), 
                **kwargs
            )
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s')
            results_dir = os.path.join(args.results_dir, 'centralized', model_type)
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            print(f'Saved results to {results_dir}/results.pkl\n')
        if 'federated' in args.settings:
            print(f'Training {model_type} in federated setting.')
            t = time.time()
            kwargs = config[model_type]['default'] | config[model_type]['federated']
            results = federated(
                seed = args.seed, 
                Server = getattr(servers, config[model_type]['default']['server_type']),
                Client = getattr(clients, config[model_type]['default']['client_type']),
                Model = getattr(models, model_type), 
                n_workers = args.n_workers, 
                **kwargs
            )
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s')
            results_dir = os.path.join(args.results_dir, 'federated', model_type)
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            print(f'Saved results to {results_dir}/results.pkl\n')
        if 'isolated' in args.settings:
            print(f'Training {model_type} in isolated setting')
            t = time.time()
            kwargs = config[model_type]['default'] | config[model_type]['isolated']
            results = isolated(
                seed = args.seed, 
                Client = getattr(clients, config[model_type]['default']['client_type']),
                Model = getattr(models, model_type), 
                n_workers = args.n_workers, 
                **kwargs
            )
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s')
            results_dir = os.path.join(args.results_dir, 'isolated', model_type)
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            print(f'Saved results to {results_dir}/results.pkl\n')

if __name__ == '__main__':
    main()

