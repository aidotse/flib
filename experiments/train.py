import argparse
from flib.train import centralized, federated, isolated
from flib.train.federated import HyperparamTuner
import pandas as pd
import os
import pickle
import time
import best_hyperparams

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--clients', nargs='+', help='Types of clients to train.', default=['MLPClient']) # LogRegClient, DecisionTreeClient, RandomForestClient, GradientBoostingClient, SVMClient, KNNClient, MLPClient
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "isolated", "centralized" or "federated".', default=['isolated']) # centralized, federated, isolated
    
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
    parser.add_argument('--testdata_files', nargs='+', help='Paths to trainsets. If one, it will be used as server testset.', default=[
        '/home/edvin/Desktop/flib/experiments/data/3_banks_homo_mid/preprocessed/a_nodes_test.csv',
        '/home/edvin/Desktop/flib/experiments/data/3_banks_homo_mid/preprocessed/b_nodes_test.csv',
        '/home/edvin/Desktop/flib/experiments/data/3_banks_homo_mid/preprocessed/c_nodes_test.csv'
    ])
    parser.add_argument('--valset_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--n_workers', type=int, help='Number of processes.', default=3)

    #parser.add_argument('--device', type=str, help='Device for computations. Can be "cpu" or cuda device, e.g. "cuda:0".', default="cuda:0")
    #parser.add_argument('--optimizer', type=str, default='SGD')
    #parser.add_argument('--criterion', type=str, default='ClassBalancedLoss') # ClassBalancedLoss, gini
    #parser.add_argument('--beta', type=float, help='Value of beta for ClassBalancedLoss.', default=0.9)
    #parser.add_argument('--n_rounds', type=int, help='Number of traning rounds.', default=3)
    #parser.add_argument('--eval_every', type=int, help='Number of rounds between evaluations.', default=1)
    #parser.add_argument('--local_epochs', type=int, help='Number of local epochs at clients.', default=1)
    #parser.add_argument('--batch_size', type=int, help='Batch size.', default=512)
    #parser.add_argument('--lr', type=float, help='Learning rate.', default=0.02)
    #parser.add_argument('--lr_patience', type=int, help='Number of epochs to wait before reducing learning rate.', default=5)
    #parser.add_argument('--es_patience', type=int, help='Number of epochs to wait before early stopping.', default=15)
    
    parser.add_argument('--results_dir', type=str, default='/home/edvin/Desktop/flib/experiments/results/3_banks_homo_mid/')
    
    args = parser.parse_args()
    
    #print(f'\nclients: {args.clients}')
    #print(f'settings: {args.settings}')    
    #print(f'traindata files:')
    #for traindata_file in args.traindata_files:
    #    print(f'    {traindata_file}')
    #print(f'valdata files:')
    #for valdata_file in args.valdata_files:
    #    print(f'    {valdata_file}')
    #print(f'testdata_files:')
    #for testdata_file in args.testdata_files:
    #    print(f'    {testdata_file}')
    #print(f'valset_size: {args.valset_size}')
    #print(f'optimizer: {args.optimizer}')
    #print(f'criterion: {args.criterion}')
    #print(f'beta: {args.beta}')
    #print(f'seed: {args.seed}')
    #print(f'n_rounds: {args.n_rounds}')
    #print(f'eval_every: {args.eval_every}')
    #print(f'batch_size: {args.batch_size}')
    #print(f'lr: {args.lr}')
    #print(f'n_workers: {args.n_workers}')
    #print(f'device: {args.device}')
    #print(f'results_dir: {args.results_dir}\n')
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    for traindata_file, valdata_file, testdata_file in zip(args.traindata_files, args.valdata_files, args.testdata_files):
        train_df = pd.read_csv(traindata_file).drop(columns=['account', 'bank'])
        if valdata_file is not None:
            val_df = pd.read_csv(valdata_file).drop(columns=['account', 'bank'])
        elif args.valset_size is not None:
            val_df = train_df.sample(frac=args.valset_size, random_state=args.seed)
            train_df = train_df.drop(val_df.index)
        else:
            val_dfs = None
        if testdata_file is not None:
            test_df = pd.read_csv(testdata_file).drop(columns=['account', 'bank'])
        else:
            test_df = None
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)
    
    for client in args.clients:
        params = getattr(best_hyperparams, f'{client}_params')
        if 'centralized' in args.settings:
            print(f'Training {client} in centralized setting.')
            t = time.time()
            results = centralized(
                seed=args.seed,
                n_workers=args.n_workers,
                train_dfs=train_dfs, 
                val_dfs=val_dfs,
                test_dfs=test_dfs,
                client=client,
                **params['centralized']
            )
            t = time.time() - t
            print('Done')
            print(f'Exec time: {t:.2f}s')
            results_dir = os.path.join(args.results_dir, 'centralized', client)
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
                pickle.dump(results, f)
            print(f'Saved results to {results_dir}/results.pkl\n')
        if 'federated' in args.settings:
            print(f'Training {client} in federated setting.')
            t = time.time()
            results = federated(
                seed=args.seed,
                n_workers=args.n_workers,
                train_dfs=train_dfs, 
                val_dfs=val_dfs,
                test_dfs=test_dfs,
                client=client,
                **params['federated']
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
            results = isolated(
                seed=args.seed,
                n_workers=args.n_workers,
                train_dfs=train_dfs, 
                val_dfs=val_dfs,
                test_dfs=test_dfs,
                client=client,
                **params['isolated']
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

