import time
import pandas as pd
import numpy as np
import os

def load_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df['bankOrig'] != 'source'] # TODO: create features based on source transactions? no real-world argument for this feature...
    df.drop(columns=['type', 'oldbalanceOrig', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'alertID', 'modelType'], inplace=True)
    return df


def get_nodes(df:pd.DataFrame, bank) -> pd.DataFrame:
    nodes = cal_node_features(df, bank)
    return nodes

def get_edges(df:pd.DataFrame, aggregated:bool=True, directional:bool=False) -> pd.DataFrame:
    if aggregated:
        edges = cal_edge_features(df, directional)
    elif not aggregated:
        edges = df[['step', 'nameOrig', 'nameDest', 'amount', 'isSAR']].rename(columns={'step': 't', 'nameOrig': 'src', 'nameDest': 'dst', 'isSAR': 'is_sar'})
    return edges


def cal_node_features(df:pd.DataFrame, bank) -> pd.DataFrame:
    
    # filter out transactions to the sink
    df_spending = df[df['bankDest'] == 'sink'].rename(columns={'nameOrig': 'account'})
    
    # filter out and reform transactions within the network 
    df_network = df[df['bankDest'] != 'sink']
    df1 = df_network[['nameOrig', 'bankOrig', 'amount', 'nameDest', 'daysInBankOrig', 'phoneChangesOrig', 'isSAR']].rename(columns={'nameOrig': 'account', 'bankOrig': 'bank', 'nameDest': 'counterpart', 'daysInBankOrig': 'days_in_bank', 'phoneChangesOrig': 'n_phone_changes', 'isSAR': 'is_sar'})
    df2 = df_network[['nameDest', 'bankDest', 'amount', 'nameOrig', 'daysInBankDest', 'phoneChangesDest', 'isSAR']].rename(columns={'nameDest': 'account', 'bankDest': 'bank', 'nameOrig': 'counterpart', 'daysInBankDest': 'days_in_bank', 'phoneChangesDest': 'n_phone_changes', 'isSAR': 'is_sar'})
    df2['amount'] = df2['amount'] * -1
    df_network = pd.concat([df1, df2])
    
    # calculate spending features
    gb = df_spending.groupby(['account'])
    sums_spending = gb['amount'].sum().rename('sum_spending')
    means_spending = gb['amount'].mean().rename('mean_spending')
    medians_spending = gb['amount'].median().rename('median_spending')
    stds_spending = gb['amount'].std().fillna(0.0).rename('std_spending')
    maxs_spending = gb['amount'].max().rename('max_spending')
    mins_spending = gb['amount'].min().rename('min_spending')
    counts_spending = gb['amount'].count().rename('count_spending')
    
    # calculate network features
    banks = df_network[['account', 'bank']].drop_duplicates().set_index('account')
    gb = df_network.groupby(['account'])
    sums = gb['amount'].sum().rename('sum') # TODO: add in and out sums 
    means = gb['amount'].mean().rename('mean')
    medians = gb['amount'].median().rename('median')
    stds = gb['amount'].std().fillna(0.0).rename('std')
    maxs = gb['amount'].max().rename('max')
    mins = gb['amount'].min().rename('min')
    counts_in = gb['amount'].apply(lambda x: (x>0).sum()).rename('count_in')
    counts_out = gb['amount'].apply(lambda x: (x<0).sum()).rename('count_out')
    counts_unique_in = gb.apply(lambda x: x[x['amount']>0]['counterpart'].nunique()).rename('count_unique_in')
    counts_unique_out = gb.apply(lambda x: x[x['amount']<0]['counterpart'].nunique()).rename('count_unique_out')
    counts_days_in_bank = gb['days_in_bank'].max().rename('count_days_in_bank')
    counts_phone_changes = gb['n_phone_changes'].max().rename('count_phone_changes')
    
    # find label
    is_sar = gb['is_sar'].max().rename('is_sar')
    
    # create final dataframe
    df_nodes = pd.concat([banks, sums, means, medians, stds, maxs, mins, counts_in, counts_out, counts_unique_in, counts_unique_out, counts_days_in_bank, counts_phone_changes, sums_spending, means_spending, medians_spending, stds_spending, maxs_spending, mins_spending, counts_spending, is_sar], axis=1)
    df_nodes = df_nodes[df_nodes['bank'] == bank]
    return df_nodes


def cal_edge_features(df:pd.DataFrame, directional:bool=False) -> pd.DataFrame:
    df = df[df['bankDest'] != 'sink']
    df = df[['step', 'nameOrig', 'nameDest', 'amount', 'isSAR']].rename(columns={'nameOrig': 'src', 'nameDest': 'dst', 'isSAR': 'is_sar'})
    if not directional:
        df[['src', 'dst']] = np.sort(df[['src', 'dst']], axis=1)
    gb = df.groupby(['src', 'dst'])
    sums = gb['amount'].sum().rename('sum')
    means = gb['amount'].mean().rename('mean')
    medians = gb['amount'].median().rename('median')
    stds = gb['amount'].std().fillna(0.0).rename('std') 
    maxs = gb['amount'].max().rename('max')
    mins = gb['amount'].min().rename('min')
    counts = gb['amount'].count().rename('count')
    is_sar = gb['is_sar'].max().rename('is_sar')
    df = pd.concat([sums, means, medians, stds, maxs, mins, counts, is_sar], axis=1)
    df.reset_index(inplace=True)          
    return df


def preprocess(path_to_tx_log:str, banks:list=['defult'], train_test_overlap:float=0.9):
    
    df = load_data(path_to_tx_log)
    overlap = train_test_overlap # overlap of training and testing data
    
    datasets = []
    for bank in banks:
        df_bank = df[(df['bankOrig'] == bank) | (df['bankDest'] == bank)]
        train_start = df_bank['step'].min()
        train_end = df_bank['step'].min() + (df_bank['step'].max() - df_bank['step'].min()) * (overlap+(1-overlap)/2)
        test_start = df_bank['step'].min() + (df_bank['step'].max() - df_bank['step'].min()) * (1-overlap)/2
        test_end = df_bank['step'].max()
        
        df_bank_train = df_bank[(df_bank['step'] >= train_start) & (df_bank['step'] <= train_end)]
        df_bank_test = df_bank[(df_bank['step'] >= test_start) & (df_bank['step'] <= test_end)]
        
        df_nodes_train = get_nodes(df_bank_train, bank)
        #df_edges_train = get_edges(df_bank_train[(df_bank_train['bankOrig'] == bank) & (df_bank_train['bankDest'] == bank)], aggregated=True, directional=False) # TODO: enable edges to/from the bank? the node features use these txs but unclear how to ceate a edge in this case, the edge can't be connected to a node with node features (could create node features based on edge txs, then the node features and edge features will look the same and some node features will be missing)
        df_nodes_test = get_nodes(df_bank_test, bank)
        #df_edges_test = get_edges(df_bank_test[(df_bank_test['bankOrig'] == bank) & (df_bank_test['bankDest'] == bank)], aggregated=True, directional=False)
        
        df_nodes_train.reset_index(inplace=True)
        #node_to_index = pd.Series(df_nodes_train.index, index=df_nodes_train['account']).to_dict()
        #df_edges_train['src'] = df_edges_train['src'].map(node_to_index) # OBS: in the csv files it looks like the edge src refers to the node two rows above the acculat node, this is due to the column head and that it starts counting at 0
        #df_edges_train['dst'] = df_edges_train['dst'].map(node_to_index)
        
        df_nodes_test.reset_index(inplace=True)
        #node_to_index = pd.Series(df_nodes_test.index, index=df_nodes_test['account']).to_dict()
        #df_edges_test['src'] = df_edges_test['src'].map(node_to_index)
        #df_edges_test['dst'] = df_edges_test['dst'].map(node_to_index)
        
        #os.makedirs(f'{preproccesed_data_folder}/{bank}/train', exist_ok=True)
        #os.makedirs(f'{preproccesed_data_folder}/{bank}/test', exist_ok=True)
        #
        #df_nodes_train.to_csv(f'{preproccesed_data_folder}/{bank}/train/nodes.csv', index=False)
        #df_edges_train.to_csv(f'{preproccesed_data_folder}/{bank}/train/edges.csv', index=False)
        #df_nodes_test.to_csv(f'{preproccesed_data_folder}/{bank}/test/nodes.csv', index=False)
        #df_edges_test.to_csv(f'{preproccesed_data_folder}/{bank}/test/edges.csv', index=False)
        datasets.append((df_nodes_train, df_nodes_test))
    
    return datasets