import pandas as pd
import numpy as np
import os
import time


def load_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df['bankOrig'] != 'source'] # TODO: create features based on source transactions
    df.drop(columns=['type', 'oldbalanceOrig', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'alertID', 'modelType'], inplace=True)
    return df


def get_nodes(df:pd.DataFrame, bank, windows) -> pd.DataFrame:
    nodes = cal_node_features(df, bank, windows)
    return nodes


def get_edges(df:pd.DataFrame, windows, aggregated:bool=True, directional:bool=False) -> pd.DataFrame:
    if aggregated:
        edges = cal_edge_features(df, directional, windows)
    elif not aggregated:
        edges = df[['step', 'nameOrig', 'nameDest', 'amount', 'isSAR']].rename(columns={'step': 't', 'nameOrig': 'src', 'nameDest': 'dst', 'isSAR': 'is_sar'})
    return edges


def cal_node_features(df:pd.DataFrame, bank, windows=1) -> pd.DataFrame:
    # calculate start and end step for each window
    if type(windows) == int:
        n_windows = windows
        start_step, end_step = df['step'].min(), df['step'].max()
        steps_per_window = (end_step - start_step) // n_windows
        windows = [(start_step + i*steps_per_window, start_step + (i+1)*steps_per_window-1) for i in range(n_windows)]
        windows[-1] = (windows[-1][0], end_step)    
        assert len(windows) == n_windows, 'The number of windows is not equal to the specified number of windows'
    elif type(windows) == tuple:
        num_windows, window_len = windows
        start_step, end_step = df['step'].min(), df['step'].max()
        if num_windows * window_len < end_step - start_step:
            raise ValueError(f'Number of windows {num_windows} and the windows length {window_len} do not allow coverage of the whole dataset. Inceasing number of windows or length of windows')
        window_overlap = (num_windows * window_len - (end_step - start_step + 1)) // (num_windows - 1)
        windows = [(start_step + i*(window_len-window_overlap), start_step + i*(window_len-window_overlap) + window_len-1) for i in range(num_windows)]
        windows[-1] = (end_step - window_len + 1, end_step)
    # filter out transactions to the sink
    df_spending = df[df['bankDest'] == 'sink'].rename(columns={'nameOrig': 'account'})
    # filter out and reform transactions within the network 
    df_network = df[df['bankDest'] != 'sink']
    df_in = df_network[['step', 'nameDest', 'bankDest', 'amount', 'nameOrig', 'daysInBankDest', 'phoneChangesDest', 'isSAR']].rename(columns={'nameDest': 'account', 'bankDest': 'bank', 'nameOrig': 'counterpart', 'daysInBankDest': 'days_in_bank', 'phoneChangesDest': 'n_phone_changes', 'isSAR': 'is_sar'})
    df_out = df_network[['step', 'nameOrig', 'bankOrig', 'amount', 'nameDest', 'daysInBankOrig', 'phoneChangesOrig', 'isSAR']].rename(columns={'nameOrig': 'account', 'bankOrig': 'bank', 'nameDest': 'counterpart', 'daysInBankOrig': 'days_in_bank', 'phoneChangesOrig': 'n_phone_changes', 'isSAR': 'is_sar'})
    
    df_nodes = pd.DataFrame()
    df_nodes = pd.concat([df_out[['account', 'bank']], df_in[['account', 'bank']]]).drop_duplicates().set_index('account')
    
    # calculate spending features
    for window in windows:
        gb = df_spending[(df_spending['step']>=window[0])&(df_spending['step']<=window[1])].groupby(['account'])
        df_nodes[f'sums_spending_{window[0]}_{window[1]}'] = gb['amount'].sum()
        df_nodes[f'means_spending_{window[0]}_{window[1]}'] = gb['amount'].mean()
        df_nodes[f'medians_spending_{window[0]}_{window[1]}'] = gb['amount'].median()
        df_nodes[f'stds_spending_{window[0]}_{window[1]}'] = gb['amount'].std()
        df_nodes[f'maxs_spending_{window[0]}_{window[1]}'] = gb['amount'].max()
        df_nodes[f'mins_spending_{window[0]}_{window[1]}'] = gb['amount'].min()
        df_nodes[f'counts_spending_{window[0]}_{window[1]}'] = gb['amount'].count()
    # calculate network features
    for window in windows:
        gb_in = df_in[(df_in['step']>=window[0])&(df_in['step']<=window[1])].groupby(['account'])
        df_nodes[f'sum_in{window[0]}_{window[1]}'] = gb_in['amount'].apply(lambda x: x[x > 0].sum())
        df_nodes[f'mean_in_{window[0]}_{window[1]}'] = gb_in['amount'].mean()
        df_nodes[f'median_in{window[0]}_{window[1]}'] = gb_in['amount'].median()
        df_nodes[f'std_in{window[0]}_{window[1]}'] = gb_in['amount'].std()
        df_nodes[f'max_in_{window[0]}_{window[1]}'] = gb_in['amount'].max()
        df_nodes[f'min_in_{window[0]}_{window[1]}'] = gb_in['amount'].min()
        df_nodes[f'count_in_{window[0]}_{window[1]}'] = gb_in['amount'].count()
        df_nodes[f'count_unique_in_{window[0]}_{window[1]}'] = gb_in['counterpart'].nunique()
        gb_out = df_out[(df_out['step']>=window[0])&(df_out['step']<=window[1])].groupby(['account'])
        df_nodes[f'sum_out{window[0]}_{window[1]}'] = gb_out['amount'].apply(lambda x: x[x > 0].sum())
        df_nodes[f'mean_out_{window[0]}_{window[1]}'] = gb_out['amount'].mean()
        df_nodes[f'median_out{window[0]}_{window[1]}'] = gb_out['amount'].median()
        df_nodes[f'std_out{window[0]}_{window[1]}'] = gb_out['amount'].std()
        df_nodes[f'max_out_{window[0]}_{window[1]}'] = gb_out['amount'].max()
        df_nodes[f'min_out_{window[0]}_{window[1]}'] = gb_out['amount'].min()
        df_nodes[f'count_out_{window[0]}_{window[1]}'] = gb_out['amount'].count()
        df_nodes[f'count_unique_out_{window[0]}_{window[1]}'] = gb_out['counterpart'].nunique()
    # calculate non window related features
    df = pd.concat([df_in[['account', 'days_in_bank', 'n_phone_changes', 'is_sar']], df_out[['account', 'days_in_bank', 'n_phone_changes', 'is_sar']]])
    gb = df.groupby('account')
    df_nodes[f'counts_days_in_bank'] = gb['days_in_bank'].max()
    df_nodes[f'counts_phone_changes'] = gb['n_phone_changes'].max()
    # find label
    df_nodes['is_sar'] = gb['is_sar'].max()
    # filter out nodes not belonging to the bank
    df_nodes = df_nodes[df_nodes['bank'] == bank] # TODO: keep these nodes? see TODO below about get edges
    # if any value is nan, there was no transaction in the window for that account and hence the feature should be 0
    df_nodes = df_nodes.fillna(0.0)
    # check if there is any missing values
    assert df_nodes.isnull().sum().sum() == 0, 'There are missing values in the node features'
    # check if there are any negative values in all comuns except the bank column
    assert (df_nodes.drop(columns='bank') < 0).sum().sum() == 0, 'There are negative values in the node features'
    return df_nodes


def cal_edge_features(df:pd.DataFrame, directional:bool=False, windows=1) -> pd.DataFrame:
    # calculate start and end step for each window
    if type(windows) == int:
        n_windows = windows
        start_step, end_step = df['step'].min(), df['step'].max()
        steps_per_window = (end_step - start_step) // n_windows
        windows = [(start_step + i*steps_per_window, start_step + (i+1)*steps_per_window-1) for i in range(n_windows)]
        windows[-1] = (windows[-1][0], end_step)
    elif type(windows) == tuple:
        num_windows, window_len = windows
        start_step, end_step = df['step'].min(), df['step'].max()
        if num_windows * window_len < end_step - start_step:
            raise ValueError(f'Number of windows {num_windows} and the windows length {window_len} do not allow coverage of the whole dataset. Inceasing number of windows or length of windows')
        window_overlap = (num_windows * window_len - (end_step - start_step + 1)) // (num_windows - 1)
        windows = [(start_step + i*(window_len-window_overlap), start_step + i*(window_len-window_overlap) + window_len-1) for i in range(num_windows)]
        windows[-1] = (end_step - window_len + 1, end_step)
    # filter out transactions to the sink
    df = df[df['bankDest'] != 'sink']
    # rename
    df = df[['step', 'nameOrig', 'nameDest', 'amount', 'isSAR']].rename(columns={'nameOrig': 'src', 'nameDest': 'dst', 'isSAR': 'is_sar'})
    # if directional=False then sort src and dst
    if not directional:
        df[['src', 'dst']] = np.sort(df[['src', 'dst']], axis=1)
    # init final dataframe
    df_edges = pd.DataFrame()
    for window in windows:
        gb = df[(df['step']>=window[0])&(df['step']<=window[1])].groupby(['src', 'dst'])
        df_edges[f'sums_{window[0]}_{window[1]}'] = gb['amount'].sum()
        df_edges[f'means_{window[0]}_{window[1]}'] = gb['amount'].mean()
        df_edges[f'medians_{window[0]}_{window[1]}'] = gb['amount'].median()
        df_edges[f'stds_{window[0]}_{window[1]}'] = gb['amount'].std().fillna(0.0)
        df_edges[f'maxs_{window[0]}_{window[1]}'] = gb['amount'].max()
        df_edges[f'mins_{window[0]}_{window[1]}'] = gb['amount'].min()
        df_edges[f'counts_{window[0]}_{window[1]}'] = gb['amount'].count()
    # find label
    gb = df.groupby(['src', 'dst'])
    df_edges[f'is_sar'] = gb['is_sar'].max()
    df_edges.reset_index(inplace=True)
    # if any value is nan, there was no transaction in the window for that edge and hence the feature should be 0
    df_edges = df_edges.fillna(0.0)
    # check if there is any missing values
    assert df_edges.isnull().sum().sum() == 0, 'There are missing values in the edge features'
    # check if there are any negative values in all comuns except the bank column
    assert (df_edges.drop(columns=['src', 'dst']) < 0).sum().sum() == 0, 'There are negative values in the edge features'
    return df_edges


def cal_features(path_to_tx_log:str, banks=None, windows:int=1, overlap:float=0.9, include_edges=True) -> list:
    df = load_data(path_to_tx_log)
    
    if banks is None:
        banks = df['bankOrig'].unique()
    
    datasets = []
    
    for bank in banks:
        df_bank = df[(df['bankOrig'] == bank) | (df['bankDest'] == bank)]
        train_start = df_bank['step'].min()
        train_end = df_bank['step'].min() + (df_bank['step'].max() - df_bank['step'].min()) * (overlap+(1-overlap)/2)
        test_start = df_bank['step'].min() + (df_bank['step'].max() - df_bank['step'].min()) * (1-overlap)/2
        test_end = df_bank['step'].max()
        df_bank_train = df_bank[(df_bank['step'] >= train_start) & (df_bank['step'] <= train_end)]
        df_bank_test = df_bank[(df_bank['step'] >= test_start) & (df_bank['step'] <= test_end)]
        df_nodes_train = get_nodes(df_bank_train, bank, windows)
        df_nodes_test = get_nodes(df_bank_test, bank, windows)
        df_nodes_train.reset_index(inplace=True)
        df_nodes_test.reset_index(inplace=True)
        
        if include_edges:
            df_edges_train = get_edges(df_bank_train[(df_bank_train['bankOrig'] == bank) & (df_bank_train['bankDest'] == bank)], windows, aggregated=True, directional=True) # TODO: enable edges to/from the bank? the node features use these txs but unclear how to ceate a edge in this case, the edge can't be connected to a node with node features (could create node features based on edge txs, then the node features and edge features will look the same and some node features will be missing)
            df_edges_test = get_edges(df_bank_test[(df_bank_test['bankOrig'] == bank) & (df_bank_test['bankDest'] == bank)], windows, aggregated=True, directional=True)
            node_to_index = pd.Series(df_nodes_train.index, index=df_nodes_train['account']).to_dict()
            df_edges_train['src'] = df_edges_train['src'].map(node_to_index) # OBS: in the csv files it looks like the edge src refers to the node two rows above the acculat node, this is due to the column head and that it starts counting at 0
            df_edges_train['dst'] = df_edges_train['dst'].map(node_to_index)
            node_to_index = pd.Series(df_nodes_test.index, index=df_nodes_test['account']).to_dict()
            df_edges_test['src'] = df_edges_test['src'].map(node_to_index)
            df_edges_test['dst'] = df_edges_test['dst'].map(node_to_index)
        else:
            df_edges_train = None
            df_edges_test = None
        
        trainset = (df_nodes_train, df_edges_train)
        testset = (df_nodes_test, df_edges_test)
        
        datasets.append((trainset, testset))
    
    return datasets
