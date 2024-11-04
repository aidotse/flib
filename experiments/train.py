import argparse
from flib.train import centralized, federated, isolated
import pandas as pd
import os
import pickle
import time
import best_hyperparams

def main():
    
    DATASET = '3_banks_homo_mid' # '30K_accts', '3_banks_homo_mid'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients', nargs='+', help='Types of clients to train.', default=['GraphSAGEClient']) # LogRegClient, DecisionTreeClient, RandomForestClient, GradientBoostingClient, SVMClient, KNNClient, MLPClient, GraphSAGEClient
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "isolated", "centralized" or "federated".', default=['centralized', 'federated', 'isolated']) # centralized, federated, isolated
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
    parser.add_argument('--test_nodes_files', nargs='+', help='Paths to test nodes files', default=[
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_nodes_test.csv',
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_nodes_test.csv',
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_nodes_test.csv'
    ])
    parser.add_argument('--test_edges_files', nargs='+', help='Paths to test edges files', default=[
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_edges_test.csv',
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_edges_test.csv',
        f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_edges_test.csv'
    ])
    parser.add_argument('--valset_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--n_workers', type=int, help='Number of processes.', default=3)
    parser.add_argument('--results_dir', type=str, default=f'/home/edvin/Desktop/flib/experiments/results/{DATASET}/')
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
    print(f'test files:')
    for test_nodes_file, test_edges_file in zip(args.test_nodes_files, args.test_edges_files):
        print(f'    {test_nodes_file}')
        print(f'    {test_edges_file}')
    print(f'valset size: {args.valset_size}')
    print(f'seed: {args.seed}')
    print(f'n workers: {args.n_workers}')
    print(f'results dir: {args.results_dir}\n')
    
    #train_dfs = []
    #val_dfs = []
    #test_dfs = []
    #for traindata_file, valdata_file, testdata_file in zip(args.train_nodes_files, args.val_nodes_files, args.test_nodes_files):
    #    train_df = pd.read_csv(traindata_file).drop(columns=['account', 'bank'])
    #    if valdata_file is not None:
    #        val_df = pd.read_csv(valdata_file).drop(columns=['account', 'bank'])
    #    elif args.valset_size is not None:
    #        val_df = train_df.sample(frac=args.valset_size, random_state=args.seed)
    #        train_df = train_df.drop(val_df.index)
    #    else:
    #        val_dfs = None
    #    if testdata_file is not None:
    #        test_df = pd.read_csv(testdata_file).drop(columns=['account', 'bank'])
    #    else:
    #        test_df = None
    #    train_dfs.append(train_df)
    #    val_dfs.append(val_df)
    #    test_dfs.append(test_df)
    
    for client in args.clients:
        params = getattr(best_hyperparams, f'{client}_params')
        if 'centralized' in args.settings:
            print(f'Training {client} in centralized setting.')
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
            c0_test_nodes_df = pd.read_csv(args.test_nodes_files[0]).drop(columns=['account', 'bank'])
            c0_test_edges_df = pd.read_csv(args.test_edges_files[0])
            c1_test_nodes_df = pd.read_csv(args.test_nodes_files[1]).drop(columns=['account', 'bank'])
            c1_test_edges_df = pd.read_csv(args.test_edges_files[1])
            c1_test_edges_df['src'] += c0_test_nodes_df.shape[0]
            c1_test_edges_df['dst'] += c0_test_nodes_df.shape[0]
            c2_test_nodes_df = pd.read_csv(args.test_nodes_files[2]).drop(columns=['account', 'bank'])
            c2_test_edges_df = pd.read_csv(args.test_edges_files[2])
            c2_test_edges_df['src'] += c0_test_nodes_df.shape[0] + c1_test_nodes_df.shape[0]
            c2_test_edges_df['dst'] += c0_test_nodes_df.shape[0] + c1_test_nodes_df.shape[0]
            test_nodes_df = pd.concat([c0_test_nodes_df, c1_test_nodes_df, c2_test_nodes_df])
            test_edges_df = pd.concat([c0_test_edges_df, c1_test_edges_df, c2_test_edges_df])
            train_df = (train_nodes_df, train_edges_df)
            test_df = (test_nodes_df, test_edges_df)
            results = centralized(
                seed=args.seed,
                n_workers=args.n_workers,
                train_dfs=[train_df], 
                #val_dfs=val_dfs,
                test_dfs=[test_df],
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
            train_dfs = [
                (pd.read_csv(args.train_nodes_files[0]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[0])),
                (pd.read_csv(args.train_nodes_files[1]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[1])),
                (pd.read_csv(args.train_nodes_files[2]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[2])),
            ]
            test_dfs = [
                (pd.read_csv(args.test_nodes_files[0]).drop(columns=['account', 'bank']), pd.read_csv(args.test_edges_files[0])),
                (pd.read_csv(args.test_nodes_files[1]).drop(columns=['account', 'bank']), pd.read_csv(args.test_edges_files[1])),
                (pd.read_csv(args.test_nodes_files[2]).drop(columns=['account', 'bank']), pd.read_csv(args.test_edges_files[2])),
            ]
            results = federated(
                seed=args.seed,
                n_workers=args.n_workers,
                train_dfs=train_dfs, 
                #val_dfs=val_dfs,
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
            train_dfs = [
                (pd.read_csv(args.train_nodes_files[0]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[0])),
                (pd.read_csv(args.train_nodes_files[1]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[1])),
                (pd.read_csv(args.train_nodes_files[2]).drop(columns=['account', 'bank']), pd.read_csv(args.train_edges_files[2])),
            ]
            test_dfs = [
                (pd.read_csv(args.test_nodes_files[0]).drop(columns=['account', 'bank']), pd.read_csv(args.test_edges_files[0])),
                (pd.read_csv(args.test_nodes_files[1]).drop(columns=['account', 'bank']), pd.read_csv(args.test_edges_files[1])),
                (pd.read_csv(args.test_nodes_files[2]).drop(columns=['account', 'bank']), pd.read_csv(args.test_edges_files[2])),
            ]
            results = isolated(
                seed=args.seed,
                n_workers=args.n_workers,
                train_dfs=train_dfs, 
                #val_dfs=val_dfs,
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

