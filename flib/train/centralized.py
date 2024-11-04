import pandas as pd
from flib.utils import set_random_seed
from flib.train import Clients
import pandas as pd
import time
import pickle
import os

def centralized(train_dfs=[], val_dfs=[], test_dfs=[], seed=42, client='LogRegClient', **kwargs):
    
    set_random_seed(seed)
    
    # concatinat dataframes
    if len(train_dfs) == 1:
        train_df = train_dfs[0]
    else:
        train_df = pd.concat(train_dfs)
    if len(val_dfs) == 1:
        val_df = val_dfs[0]
    else:
        val_dfs = [val_df for val_df in val_dfs if val_df is not None]
        if val_dfs:
            val_df = pd.concat(val_dfs)
        else:
            val_df = None
    if len(test_dfs) == 1:
        test_df = test_dfs[0]
    else:
        test_dfs = [test_df for test_df in test_dfs if test_df is not None]
        if test_dfs:
            test_df = pd.concat(test_dfs)
        else:
            test_df = None
    
    Client = getattr(Clients, client)
    
    client = Client(
        name=f'c0',
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        **kwargs
    )
    
    results = {client.name:  client.run(**kwargs)}
    return results

if __name__ == '__main__':
    
    DATASET = '3_banks_homo_mid' # '30K_accts', '3_banks_homo_mid'
    
    # for debugging
    c0_train_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_nodes_train.csv'
    c0_train_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_edges_train.csv'
    c1_train_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_nodes_train.csv'
    c1_train_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_edges_train.csv'
    c2_train_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_nodes_train.csv'
    c2_train_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_edges_train.csv'
    c0_test_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_nodes_test.csv'
    c0_test_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c0_edges_test.csv'
    c1_test_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_nodes_test.csv'
    c1_test_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c1_edges_test.csv'
    c2_test_nodes_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_nodes_test.csv'
    c2_test_edges_csv = f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/preprocessed/c2_edges_test.csv'
    
    c0_train_nodes_df = pd.read_csv(c0_train_nodes_csv).drop(columns=['account', 'bank'])
    c0_train_edges_df = pd.read_csv(c0_train_edges_csv)
    
    c1_train_nodes_df = pd.read_csv(c1_train_nodes_csv).drop(columns=['account', 'bank'])
    c1_train_edges_df = pd.read_csv(c1_train_edges_csv)
    c1_train_edges_df['src'] += c0_train_nodes_df.shape[0]
    c1_train_edges_df['dst'] += c0_train_nodes_df.shape[0]
    
    c2_train_nodes_df = pd.read_csv(c2_train_nodes_csv).drop(columns=['account', 'bank'])
    c2_train_edges_df = pd.read_csv(c2_train_edges_csv)
    c2_train_edges_df['src'] += c0_train_nodes_df.shape[0] + c1_train_nodes_df.shape[0]
    c2_train_edges_df['dst'] += c0_train_nodes_df.shape[0] + c1_train_nodes_df.shape[0]
    
    train_nodes_df = pd.concat([c0_train_nodes_df, c1_train_nodes_df, c2_train_nodes_df])
    train_edges_df = pd.concat([c0_train_edges_df, c1_train_edges_df, c2_train_edges_df])
    
    c0_test_nodes_df = pd.read_csv(c0_test_nodes_csv).drop(columns=['account', 'bank'])
    c0_test_edges_df = pd.read_csv(c0_test_edges_csv)
    
    c1_test_nodes_df = pd.read_csv(c1_test_nodes_csv).drop(columns=['account', 'bank'])
    c1_test_edges_df = pd.read_csv(c1_test_edges_csv)
    c1_test_edges_df['src'] += c0_test_nodes_df.shape[0]
    c1_test_edges_df['dst'] += c0_test_nodes_df.shape[0]
    
    c2_test_nodes_df = pd.read_csv(c2_test_nodes_csv).drop(columns=['account', 'bank'])
    c2_test_edges_df = pd.read_csv(c2_test_edges_csv)
    c2_test_edges_df['src'] += c0_test_nodes_df.shape[0] + c1_test_nodes_df.shape[0]
    c2_test_edges_df['dst'] += c0_test_nodes_df.shape[0] + c1_test_nodes_df.shape[0]
    
    test_nodes_df = pd.concat([c0_test_nodes_df, c1_test_nodes_df, c2_test_nodes_df])
    test_edges_df = pd.concat([c0_test_edges_df, c1_test_edges_df, c2_test_edges_df])
    
    train_df = (train_nodes_df, train_edges_df)
    test_df = (test_nodes_df, test_edges_df)
    
    params = {
        'rounds': 100,
        'eval_every': 10,
        'lr_patience': 100,
        'es_patience': 100,
        'device': 'cuda:0',
        'batch_size': 128,
        'hidden_dim': 64,
        'optimizer': 'Adam',
        'optimizer_params': {
            'lr': 0.01,
            'weight_decay': 0.0,
            'amsgrad': False,
        },
        'criterion': 'CrossEntropyLoss',
        'criterion_params': {}
    }
    
    t = time.time()
    results = centralized(train_dfs=[train_df], test_dfs=[test_df], seed=42, n_workers=3, client='GraphSAGEClient', **params)
    t = time.time() - t
    print('Done')
    print(f'Exec time: {t:.2f}s')
    results_dir = f'/home/edvin/Desktop/flib/experiments/results/{DATASET}/centralized/GraphSAGEClient'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved results to {results_dir}/results.pkl\n')
    