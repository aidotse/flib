import argparse
from flib.train import centralized, federated, isolated
import pandas as pd
import os
import pickle
import time
import yaml

def main():
    
    EXPERIMENT = '3_banks_homo_easy' # '30K_accts', '3_banks_homo_mid'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_config_files', nargs='+', help='Paths to client config files.', default=[
        f'experiments/{EXPERIMENT}/clients/c0/config.yaml',
        f'experiments/{EXPERIMENT}/clients/c1/config.yaml',
        f'experiments/{EXPERIMENT}/clients/c2/config.yaml'
    ])
    parser.add_argument('--centralized_config_file', type=str, help='Path to centralised config file.', default=f'experiments/{EXPERIMENT}/data/config.yaml')
    parser.add_argument('--clients', nargs='+', help='Client types to train.', default=['LogRegClient']) # LogRegClient, DecisionTreeClient, RandomForestClient, GradientBoostingClient, SVMClient, KNNClient, MLPClient, GraphSAGE
    parser.add_argument('--n_workers', type=int, help='Number of processes.', default=3)
    parser.add_argument('--results_dir', type=str, default=f'experiments/{EXPERIMENT}/results')
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "isolated", "centralized" or "federated".', default=['centralized', 'federated', 'isolated']) # centralized, federated, isolated
    args = parser.parse_args()
    
    print('\nParsed arguments:')
    for arg, val in vars(args).items():
        print(f'{arg}: {val}')
    
    for client in args.clients:
        if 'centralized' in args.settings:
            print(f'\nTraining {client} in centralized setting.')
            t = time.time()
            with open(args.centralized_config_file, 'r') as f:
                config = yaml.safe_load(f)
            for file, path in config['data'].items():
                if not os.path.isabs(path):
                    config['data'][file] = os.path.join(os.path.dirname(args.centralized_config_file), path)
            client_params = config['data'] | config['hyperparameters'][client]['default']
            results = centralized(
                seed=args.seed,
                n_workers=args.n_workers,
                client_type=client,
                client_names=config['preprocess']['bank'],
                client_params=client_params
            )
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s')
            results_dir = os.path.join(args.results_dir, 'centralized', client)
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            print(f'Saved results to {results_dir}/results.pkl\n')
        if 'federated' in args.settings and not client == 'DecisionTreeClient':
            print(f'Training {client} in federated setting.')
            t = time.time()
            client_type = client
            client_names = []
            client_params = []
            for config_file in args.client_config_files:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                for data_file, path in config['data'].items():
                    if not os.path.isabs(path):
                        config['data'][data_file] = os.path.join(os.path.dirname(config_file), path)
                params = config['data'] | config['hyperparameters'][client_type]['default']
                client_names.append(config['preprocess']['bank'])
                client_params.append(params)
            results = federated(
                seed=args.seed,
                n_workers=args.n_workers,
                client_type=client_type,
                client_names=client_names,
                client_params=client_params
            )
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s')
            results_dir = os.path.join(args.results_dir, 'federated', client)
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            print(f'Saved results to {results_dir}/results.pkl\n')
        if 'isolated' in args.settings:
            print(f'Training {client} in isolated setting')
            t = time.time()
            client_type = client
            client_names = []
            client_params = []
            for config_file in args.client_config_files:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                for data_file, path in config['data'].items():
                    if not os.path.isabs(path):
                        config['data'][data_file] = os.path.join(os.path.dirname(config_file), path)
                params = config['data'] | config['hyperparameters'][client_type]['default']
                client_names.append(config['preprocess']['bank'])
                client_params.append(params)
            results = isolated(
                seed=args.seed,
                n_workers=args.n_workers,
                client_type=client_type,
                client_names=client_names,
                client_params=client_params,
            )
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s')
            results_dir = os.path.join(args.results_dir, 'isolated', client)
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            print(f'Saved results to {results_dir}/results.pkl\n')

if __name__ == '__main__':
    main()

