import argparse
import os
import time

import pandas as pd

from flib.train import centralized, federated, isolated, HyperparamTuner
import hyperparams

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--clients', nargs='+', help='Types of clients to train.', default=['LogRegClient', 'DecisionTreeClient', 'RandomForestClient', 'GradientBoostingClient', 'SVMClient', 'KNNClient']) # LogRegClient, DecisionTreeClient, RandomForestClient, GradientBoostingClient, SVMClient, KNNClient
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "isolated", "centralized" or "federated".', default=['centralized', 'federated', 'isolated'])
    parser.add_argument('--traindata_files', nargs='+', help='Paths to trainsets.', default=[
        '/home/edvin/Desktop/flib/experiments/data/3_banks_homo_mid/preprocessed/a_nodes_train.csv',
        '/home/edvin/Desktop/flib/experiments/data/3_banks_homo_mid/preprocessed/b_nodes_train.csv',
        '/home/edvin/Desktop/flib/experiments/data/3_banks_homo_mid/preprocessed/c_nodes_train.csv'
    ])
    parser.add_argument('--valdata_files', nargs='+', help='Paths to valsets', default=[
        None,
        None,
        None
    ])
    parser.add_argument('--valset_size', type=float, default=0.2)
    
    parser.add_argument('--n_workers', type=int, help='Number of processes.', default=3)
    parser.add_argument('--device', type=str, help='Device for computations. Can be "cpu" or cuda device, e.g. "cuda:0".', default="cuda:0")
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--n_trials', type=int, help='Number of trials.', default=10)
    parser.add_argument('--results_dir', type=str, default='/home/edvin/Desktop/flib/experiments/results/3_banks_homo_mid/')
    
    args = parser.parse_args()
    
    print()
    print(f'clients: {args.clients}')
    print(f'settings: {args.settings}')    
    print(f'traindata files:')
    for traindata_file in args.traindata_files:
        print(f'    {traindata_file}')
    print(f'valdata files:')
    for valdata_file in args.valdata_files:
        print(f'    {valdata_file}')
    print(f'valset_size: {args.valset_size}')
    print(f'n_workers: {args.n_workers}')
    print(f'device: {args.device}')
    print(f'results_dir: {args.results_dir}')
    print(f'n_trials: {args.n_trials}')
    print()
    
    train_dfs = []
    val_dfs = []
    for traindata_file, valdata_file in zip(args.traindata_files, args.valdata_files):
        train_df = pd.read_csv(traindata_file).drop(columns=['account', 'bank'])
        if valdata_file is not None:
            val_df = pd.read_csv(valdata_file).drop(columns=['account', 'bank'])
        elif args.valset_size is not None:
            val_df = train_df.sample(frac=args.valset_size, random_state=args.seed)
            train_df = train_df.drop(val_df.index)
        else:
            val_dfs = None
        train_dfs.append(train_df)
        val_dfs.append(val_df)
    
    for client in args.clients:
        if 'centralized' in args.settings:
            print(f'\nTurning hyperparameters for {client} in a centralized setting.')
            t = time.time()
            study_name = f'{client}_centralized'
            os.makedirs(os.path.join(args.results_dir, f'centralized/{client}'), exist_ok=True)
            storage = 'sqlite:///' + os.path.join(args.results_dir, f'centralized/{client}/hp_study.db')
            hyperparamtuner = HyperparamTuner(
                study_name=study_name,
                obj_fn=centralized,
                train_dfs=train_dfs,
                val_dfs=val_dfs,
                seed=args.seed,
                storage=storage,
                client=client,
                n_workers = args.n_workers,
                device = args.device,
                params = getattr(hyperparams, f'{client}_params')
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
        
        if 'federated' in args.settings and client == 'LogRegClient':
            print(f'\nTurning hyperparameters for {client} in a federated setting.')
            t = time.time()
            study_name = f'{client}_federated'
            os.makedirs(os.path.join(args.results_dir, f'federated/{client}'), exist_ok=True)
            storage = 'sqlite:///' + os.path.join(args.results_dir, f'federated/{client}/hp_study.db')
            hyperparamtuner = HyperparamTuner(
                study_name=study_name,
                obj_fn=federated,
                train_dfs=train_dfs,
                val_dfs=val_dfs,
                seed=args.seed,
                n_workers=args.n_workers,
                device = args.device,
                storage=storage,
                client=client,
                params = getattr(hyperparams, f'{client}_params')
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
            for i, (train_df, val_df) in enumerate(zip(train_dfs, val_dfs)):
                study_name = f'{client}_isolated'
                os.makedirs(os.path.join(args.results_dir, f'isolated/{client}/c{i}'), exist_ok=True)
                storage = 'sqlite:///' + os.path.join(args.results_dir, f'isolated/{client}/c{i}/hp_study.db')
                hyperparamtuner = HyperparamTuner(
                    study_name=study_name,
                    obj_fn=isolated,
                    train_dfs=[train_df],
                    val_dfs=[val_df],
                    seed=args.seed,
                    storage=storage,
                    client=client,
                    n_workers = args.n_workers,
                    device = args.device,
                    params = getattr(hyperparams, f'{client}_params')
                )
                best_trials = hyperparamtuner.optimize(n_trials=args.n_trials)
                
                best_trials_file = os.path.join(args.results_dir, f'isolated/{client}/c{i}/best_trials.txt')
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

