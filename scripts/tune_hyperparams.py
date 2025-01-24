import argparse
import os
import time
import yaml

import pandas as pd

import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from flib.train import centralized, federated, isolated, HyperparamTuner

def main():
    
    EXPERIMENT = '3_banks_homo_mid'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_config_files', nargs='+', help='Paths to client config files.', default=[
        f'experiments/{EXPERIMENT}/clients/c0/config.yaml',
        f'experiments/{EXPERIMENT}/clients/c1/config.yaml',
        f'experiments/{EXPERIMENT}/clients/c2/config.yaml'
    ])
    parser.add_argument('--centralized_config_file', type=str, help='Path to centralised config file.', default=f'experiments/{EXPERIMENT}/data/config.yaml')
    parser.add_argument('--clients', nargs='+', help='Client types to train.', default=['LogRegClient']) # LogRegClient, DecisionTreeClient, RandomForestClient, GradientBoostingClient, SVMClient, KNNClient, MLPClient, GraphSAGE
    parser.add_argument('--n_workers', type=int, help='Number of processes.', default=3)
    parser.add_argument('--n_trials', type=int, help='Number of trials.', default=2)
    parser.add_argument('--results_dir', type=str, default=f'experiments/{EXPERIMENT}/results')
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "isolated", "centralized" or "federated".', default=['centralized', 'federated', 'isolated']) # centralized, federated, isolated
    args = parser.parse_args()
    
    print('\nParsed arguments:')
    for arg, val in vars(args).items():
        print(f'{arg}: {val}')
    
    for client in args.clients:
        if 'centralized' in args.settings:
            print(f'\nTurning hyperparameters for {client} in a centralized setting.')
            t = time.time()
            study_name = f'{client}_centralized'
            os.makedirs(os.path.join(args.results_dir, f'centralized/{client}'), exist_ok=True)
            storage = 'sqlite:///' + os.path.join(args.results_dir, f'centralized/{client}/hp_study.db')
            with open(args.centralized_config_file, 'r') as f:
                config = yaml.safe_load(f)
            for file, path in config['data'].items():
                if not os.path.isabs(path):
                    config['data'][file] = os.path.join(os.path.dirname(args.centralized_config_file), path)
            params = config['hyperparameters'][client]
            params['data'] = config['data']
            hyperparamtuner = HyperparamTuner(
                study_name=study_name,
                obj_fn=centralized,
                seed=args.seed,
                n_workers = args.n_workers,
                storage=storage,
                client_type=client,
                client_names=['cen'],
                client_data=[config['data']],
                client_params = params
            )
            best_trials = hyperparamtuner.optimize(n_trials=args.n_trials)
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s')
            best_trials_file = os.path.join(args.results_dir, f'centralized/{client}/best_trials.txt')
            with open(best_trials_file, 'w') as f:
                for trial in best_trials:
                    print(f'\ntrial: {trial.number}')
                    f.write(f'\ntrial: {trial.number}\n')
                    print(f'values: {trial.values}')
                    f.write(f'values: {trial.values}\n')
                    for param in trial.params:
                        f.write(f'{param}: {trial.params[param]}\n')
                        print(f'{param}: {trial.params[param]}')
            print()
        
        if 'federated' in args.settings and (client == 'LogRegClient' or client == 'MLPClient' or client == 'GraphSAGEClient'):
            print(f'\nTurning hyperparameters for {client} in a federated setting.')
            t = time.time()
            study_name = f'{client}_federated'
            os.makedirs(os.path.join(args.results_dir, f'federated/{client}'), exist_ok=True)
            storage = 'sqlite:///' + os.path.join(args.results_dir, f'federated/{client}/hp_study.db')
            client_names = []
            client_data = []
            for config_file in args.client_config_files:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                for data_file, path in config['data'].items():
                    if not os.path.isabs(path):
                        config['data'][data_file] = os.path.join(os.path.dirname(config_file), path)
                client_names.append(config['preprocess']['bank'])
                client_data.append(config['data'])
            params = config['hyperparameters'][client] # OBS: using params defined in the last client config file, change to the cen config file or make a fed config file?
            hyperparamtuner = HyperparamTuner(
                study_name=study_name,
                obj_fn=federated,
                seed=args.seed,
                n_workers=args.n_workers,
                storage=storage,
                client_type=client,
                client_names=client_names,
                client_data=client_data,
                client_params = params
            )
            best_trials = hyperparamtuner.optimize(n_trials=args.n_trials)
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s')
            best_trials_file = os.path.join(args.results_dir, f'federated/{client}/best_trials.txt')
            with open(best_trials_file, 'w') as f:
                for trial in best_trials:
                    print(f'\ntrial: {trial.number}')
                    f.write(f'\ntrial: {trial.number}\n')
                    print(f'values: {trial.values}')
                    f.write(f'values: {trial.values}\n')
                    for param in trial.params:
                        f.write(f'{param}: {trial.params[param]}\n')
                        print(f'{param}: {trial.params[param]}')
            print()
            
        if 'isolated' in args.settings:
            print(f'\nTurning hyperparameters for {client} in a isolated setting.')
            t = time.time()
            client_names = []
            client_data = []
            client_params = []
            for config_file in args.client_config_files:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                for data_file, path in config['data'].items():
                    if not os.path.isabs(path):
                        config['data'][data_file] = os.path.join(os.path.dirname(config_file), path)
                client_names.append(config['preprocess']['bank'])
                client_data.append(config['data'])
                client_params.append(config['hyperparameters'][client])
            for name, data, params in zip(client_names, client_data, client_params):
                study_name = f'{client}_isolated'
                os.makedirs(os.path.join(args.results_dir, f'isolated/{client}/{name}'), exist_ok=True)
                storage = 'sqlite:///' + os.path.join(args.results_dir, f'isolated/{client}/{name}/hp_study.db')
                hyperparamtuner = HyperparamTuner(
                    study_name=study_name,
                    obj_fn=isolated,
                    seed=args.seed,
                    n_workers = args.n_workers,
                    storage=storage,
                    client_type=client,
                    client_names=[name],
                    client_data=[data],
                    client_params=params,
                )
                best_trials = hyperparamtuner.optimize(n_trials=args.n_trials)
                best_trials_file = os.path.join(args.results_dir, f'isolated/{client}/{name}/best_trials.txt')
                with open(best_trials_file, 'w') as f:
                    for trial in best_trials:
                        print(f'\ntrial: {trial.number}')
                        f.write(f'\ntrial: {trial.number}\n')
                        print(f'values: {trial.values}')
                        f.write(f'values: {trial.values}\n')
                        for param in trial.params:
                            f.write(f'{param}: {trial.params[param]}\n')
                            print(f'{param}: {trial.params[param]}')
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s\n')

if __name__ == '__main__':
    main()

