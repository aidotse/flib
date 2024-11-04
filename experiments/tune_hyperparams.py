import argparse
import os
import time

import pandas as pd

import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from flib.train import centralized, federated, isolated, HyperparamTuner
import hyperparams

def main():
    
    DATASET = '3_banks_homo_mid' # '30K_accts', '3_banks_homo_mid'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients', nargs='+', help='Types of clients to train.', default=['DecisionTreeClient']) # LogRegClient, MLPClient, GraphSAGEClient, DecisionTreeClient, RandomForestClient, GradientBoostingClient, SVMClient, KNNClient
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "isolated", "centralized" or "federated".', default=['isolated']) # 'centralized', 'federated', 'isolated'
    parser.add_argument('--train_nodes_files', nargs='+', help='Paths to train nodes data.', default=[
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_nodes_train.csv',
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_nodes_train.csv',
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_nodes_train.csv'
    ])
    parser.add_argument('--train_edges_files', nargs='+', help='Paths to train edges data.', default=[
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_edges_train.csv',
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_edges_train.csv',
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_edges_train.csv'
    ])
    parser.add_argument('--val_nodes_files', nargs='+', help='Paths to val nodes data', default=[
        None,
        None,
        None
    ])
    parser.add_argument('--val_edges_files', nargs='+', help='Paths to val edges data', default=[
        None,
        None,
        None
    ])
    parser.add_argument('--valset_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--n_workers', type=int, help='Number of processes.', default=3)
    parser.add_argument('--device', type=str, help='Device for computations. Can be "cpu" or cuda device, e.g. "cuda:0".', default="cuda:0")
    parser.add_argument('--n_trials', type=int, help='Number of trials.', default=50)
    parser.add_argument('--results_dir', type=str, default=f'/home/edvin/Desktop/flib/experiments/data/{DATASET}')
    args = parser.parse_args()
    
    print(f'\nclients: {args.clients}')
    print(f'settings: {args.settings}')    
    print(f'train files:')
    for train_nodes_file, train_edges_file in zip(args.train_nodes_files, args.train_edges_files):
        print(f'    {train_nodes_file}')
        print(f'    {train_edges_file}')
    print(f'val files:')
    for val_nodes_file, val_edges_file in zip(args.val_nodes_files, args.val_edges_files):
        print(f'    {val_nodes_file}')
        print(f'    {val_edges_file}')
    print(f'valset size: {args.valset_size}')
    print(f'n workers: {args.n_workers}')
    print(f'device: {args.device}')
    print(f'n trials: {args.n_trials}')
    print(f'results dir: {args.results_dir}\n')
    
    train_dfs = []
    val_dfs = []
    for traindata_file, valdata_file in zip(args.train_nodes_files, args.val_nodes_files):
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
            c0_train_nodes_df = pd.read_csv(args.train_nodes_files[0]).drop(columns=['account', 'bank'])
            c0_train_edges_df = pd.read_csv(args.train_edges_files[0])
            c1_train_nodes_df = pd.read_csv(args.train_nodes_files[1]).drop(columns=['account', 'bank'])
            c1_train_edges_df = pd.read_csv(args.train_edges_files[1])
            c1_train_edges_df['src'] += c0_train_nodes_df.shape[0]
            c1_train_edges_df['dst'] += c0_train_nodes_df.shape[0]
            c2_train_nodes_df = pd.read_csv(args.train_nodes_files[2]).drop(columns=['account', 'bank'])
            c2_train_edges_df = pd.read_csv(args.train_edges_files[2])
            c2_train_edges_df['src'] += c0_train_nodes_df.shape[0] + c1_train_nodes_df.shape[0]
            c2_train_edges_df['dst'] += c0_train_nodes_df.shape[0] + c1_train_nodes_df.shape[0]
            train_nodes_df = pd.concat([c0_train_nodes_df, c1_train_nodes_df, c2_train_nodes_df])
            train_edges_df = pd.concat([c0_train_edges_df, c1_train_edges_df, c2_train_edges_df])
            train_df = (train_nodes_df, train_edges_df)
            study_name = f'{client}_centralized'
            os.makedirs(os.path.join(args.results_dir, f'centralized/{client}'), exist_ok=True)
            storage = 'sqlite:///' + os.path.join(args.results_dir, f'centralized/{client}/hp_study.db')
            hyperparamtuner = HyperparamTuner(
                study_name=study_name,
                obj_fn=centralized,
                train_dfs=[train_df],
                val_dfs=[None],
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
        
        if 'federated' in args.settings and (client == 'LogRegClient' or client == 'MLPClient' or client == 'GraphSAGEClient'):
            print(f'\nTurning hyperparameters for {client} in a federated setting.')
            t = time.time()
            train_dfs = [
                (pd.read_csv(args.train_nodes_files[0]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[0])),
                (pd.read_csv(args.train_nodes_files[1]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[1])),
                (pd.read_csv(args.train_nodes_files[2]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[2])),
            ]
            study_name = f'{client}_federated'
            os.makedirs(os.path.join(args.results_dir, f'federated/{client}'), exist_ok=True)
            storage = 'sqlite:///' + os.path.join(args.results_dir, f'federated/{client}/hp_study.db')
            hyperparamtuner = HyperparamTuner(
                study_name=study_name,
                obj_fn=federated,
                train_dfs=train_dfs,
                val_dfs=[None],
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
            #train_dfs = [
            #    (pd.read_csv(args.train_nodes_files[0]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[0])),
            #    (pd.read_csv(args.train_nodes_files[1]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[1])),
            #    (pd.read_csv(args.train_nodes_files[2]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[2])),
            #]
            #val_dfs = [None, None, None]
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

