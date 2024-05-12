import pandas as pd
import numpy as np
import os
import multiprocessing as mp
import time

def load_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df['bankOrig'] != 'source'] # TODO: create features based on source transactions
    #df = df[df['bankDest'] != 'sink']
    df.drop(columns=['type', 'oldbalanceOrig', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'alertID', 'modelType'], inplace=True)
    return df

def split_and_reform(df:pd.DataFrame, bank:str) -> pd.DataFrame:
    df1 = df[df['bankOrig'] == bank]
    df1 = df1.drop(columns=['bankOrig', 'bankDest'])
    df1.rename(columns={'nameOrig': 'account', 'nameDest': 'counterpart', 'isSAR': 'is_sar'}, inplace=True)
    
    df2 = df[df['bankDest'] == bank]
    df2 = df2.drop(columns=['bankOrig', 'bankDest'])
    df2.rename(columns={'nameDest': 'account', 'nameOrig': 'counterpart', 'isSAR': 'is_sar'}, inplace=True)
    
    df2['amount'] = df2['amount'] * -1
    return pd.concat([df1, df2])

def get_nodes_and_edges(df:pd.DataFrame, bank:str) -> pd.DataFrame:
    df1 = df[df['bankOrig'] == bank]
    df1 = df1.drop(columns=['bankOrig', 'bankDest'])
    df1.rename(columns={'nameOrig': 'account', 'nameDest': 'counterpart', 'isSAR': 'is_sar'}, inplace=True)
    df2 = df[df['bankDest'] == bank]
    df2 = df2.drop(columns=['bankOrig', 'bankDest'])
    df2.rename(columns={'nameDest': 'account', 'nameOrig': 'counterpart', 'isSAR': 'is_sar'}, inplace=True)
    df3 = pd.concat([df1, df2])
    df_nodes = df3.groupby('account')['is_sar'].max().to_frame().reset_index(drop=True)
    df_nodes.rename(columns={'account': 'node_id', 'is_sar': 'y'}, inplace=True)
    df_edges = df[['nameOrig', 'nameDest', 'amount']].rename(columns={'nameOrig': 'src', 'nameDest': 'dst', 'amount': 'x1'})
    return df_nodes, df_edges

def compare(input:tuple) -> tuple:
    name, df = input
    n_rows = df.shape[0]
    columns = df.columns[1:].to_list()
    anomalies = {column: 0.0 for column in columns}
    for column in columns:
        for row in range(n_rows):
            value = df.iloc[row, :][column]
            df_tmp = df.drop(df.index[row])
            tenth_percentile = df_tmp[column].quantile(0.05)
            ninetieth_percentile = df_tmp[column].quantile(0.95)
            if value < tenth_percentile or value > ninetieth_percentile:
                anomalies[column] += 1 / n_rows
    return name[0], anomalies

def compare_mp(df:pd.DataFrame, n_workers:int=mp.cpu_count()) -> list[tuple]:
    dfs = list(df.groupby(['account']))
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(compare, dfs)
    return results

def cal_spending_behavior(df:pd.DataFrame, range:list=None, interval:int=7) -> pd.DataFrame:
    if range:
        df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    df = df.loc[df['counterpart']==-2]
    df['interval_group'] = df['step'] // interval
    df['amount'] = df['amount'].abs()
    gb = df.groupby(['account', 'interval_group'])
    df_bundled = pd.concat([gb['amount'].sum().rename('volume'), gb['amount'].count().rename('count')], axis=1).reset_index().drop(columns=['interval_group'])
    list_spending_behavior = compare_mp(df_bundled)
    list_spending_behavior = [(name, d['volume'], d['count']) for name, d in list_spending_behavior]
    df_speding_behavior = pd.DataFrame(list_spending_behavior, columns=['account', 'volume', 'count'])
    return df_speding_behavior

def get_nodes(df:pd.DataFrame, bank) -> pd.DataFrame:
    nodes = cal_node_features(df, bank)
    return nodes

def get_edges(df:pd.DataFrame, aggregated:bool=True, directional:bool=False) -> pd.DataFrame:
    if aggregated:
        edges = cal_edge_features(df, directional)
    elif not aggregated:
        edges = df[['step', 'nameOrig', 'nameDest', 'amount', 'isSAR']].rename(columns={'step': 't', 'nameOrig': 'src', 'nameDest': 'dst', 'isSAR': 'is_sar'})
    return edges


def cal_node_features_without_SAR(df:pd.DataFrame, bank) -> pd.DataFrame:
    # filter out SAR transactions to the sink
    df_spending = df[(df['bankDest'] == 'sink') & (df['isSAR'] == 0)].rename(columns={'nameOrig': 'account'})

    # filter out and reform transactions within the network
    df_network = df[df['bankDest'] != 'sink' & (df['isSAR'] == 0)]
    df1 = df_network[['nameOrig', 'bankOrig', 'amount', 'nameDest', 'daysInBankOrig', 'phoneChangesOrig']].rename(columns={'nameOrig': 'account', 'bankOrig': 'bank', 'nameDest': 'counterpart', 'daysInBankOrig': 'days_in_bank', 'phoneChangesOrig': 'n_phone_changes'})
    df2 = df_network[['nameDest', 'bankDest', 'amount', 'nameOrig', 'daysInBankDest', 'phoneChangesDest']].rename(columns={'nameDest': 'account', 'bankDest': 'bank', 'nameOrig': 'counterpart', 'daysInBankDest': 'days_in_bank', 'phoneChangesDest': 'n_phone_changes'})
    df2['amount'] = df2['amount'] * -1
    df_network = pd.concat([df1, df2])
    


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
    sums = gb['amount'].sum().rename('sum')
    means = gb['amount'].mean().rename('mean')
    medians = gb['amount'].median().rename('median')
    stds = gb['amount'].std().fillna(0.0).rename('std')
    maxs = gb['amount'].max().rename('max')
    mins = gb['amount'].min().rename('min')

    in_sum=gb['amount'].apply(lambda x: x[x>0].sum()).rename('in_sum')
    out_sum=gb['amount'].apply(lambda x: x[x<0].sum()).rename('out_sum')
    in_mean=gb['amount'].apply(lambda x: x[x>0].mean()).rename('in_mean')
    out_mean=gb['amount'].apply(lambda x: x[x<0].mean()).rename('out_mean')
    in_median=gb['amount'].apply(lambda x: x[x>0].median()).rename('in_median')
    out_median=gb['amount'].apply(lambda x: x[x<0].median()).rename('out_median')
    in_std=gb['amount'].apply(lambda x: x[x>0].std()).rename('in_std')
    out_std=gb['amount'].apply(lambda x: x[x<0].std()).rename('out_std')
    in_max=gb['amount'].apply(lambda x: x[x>0].max()).rename('in_max')
    out_max=gb['amount'].apply(lambda x: x[x<0].max()).rename('out_max')
    in_min=gb['amount'].apply(lambda x: x[x>0].min()).rename('in_min')
    out_min=gb['amount'].apply(lambda x: x[x<0].min()).rename('out_min')

    count_in = gb['amount'].apply(lambda x: (x>0).sum()).rename('count_in')
    count_out = gb['amount'].apply(lambda x: (x<0).sum()).rename('count_out')
    # count_unique_in = gb.apply(lambda x: x[x['amount']>0]['counterpart'].nunique()).rename('count_unique_in')
    # count_unique_out = gb.apply(lambda x: x[x['amount']<0]['counterpart'].nunique()).rename('count_unique_out')
    # n_unique_in = gb.apply(lambda x: x[x['amount']>0]['counterpart'].nunique()).rename('n_unique_in')
    # n_unique_out = gb.apply(lambda x: x[x['amount']<0]['counterpart'].nunique()).rename('n_unique_out')
   
    count_days_in_bank = gb['days_in_bank'].max().rename('count_days_in_bank')
    count_phone_changes = gb['n_phone_changes'].max().rename('count_phone_changes')
    
    
    #For all nan values in the columns, fill with 0
    #df_nodes = df_nodes.fillna(0)
    #df_nodes = df_nodes.fillna(0)



    # find label
    is_sar = gb['is_sar'].max().rename('is_sar')
    
    # create final dataframe
    #df_nodes = pd.concat([banks, sums, means, medians, stds, maxs, mins, counts_in, counts_out, counts_unique_in, counts_unique_out, counts_days_in_bank, counts_phone_changes, sums_spending, means_spending, medians_spending, stds_spending, maxs_spending, mins_spending, counts_spending, is_sar], axis=1)
    #df_nodes = pd.concat([banks, in_sum, out_sum, in_mean, out_mean, in_median, out_median, in_std, out_std, in_max, out_max, in_min, out_min, n_unique_in, n_unique_out, count_days_in_bank, count_phone_changes, sums_spending, means_spending, medians_spending, stds_spending, maxs_spending, mins_spending, counts_spending, is_sar], axis=1)
    df_nodes = pd.concat([banks, sums, means, medians, stds, maxs, mins, in_sum, out_sum, in_mean, out_mean, in_median, out_median, in_std, out_std, in_max, out_max, in_min, out_min, count_in, count_out, count_days_in_bank, count_phone_changes, sums_spending, means_spending, medians_spending, stds_spending, maxs_spending, mins_spending, counts_spending, is_sar], axis=1)
    df_nodes = df_nodes[df_nodes['bank'] == bank]
    df_nodes.fillna(0, inplace=True)

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

def main():
    
    t = time.time()
    
    DATASET = '10K_acct_MID5_withoutSAR'
    
    #path = f'../AMLsim/outputs/{DATASET}/tx_log.csv'
    #df = load_data(path)
    print('Loading data...')
    # df = load_data('/home/agnes/desktop/flib/AMLsim/outputs/100K_accts_MID5/tx_log.csv')
    df = load_data('/home/agnes/desktop/flib/AMLsim/outputs/10K_accts_MID5/tx_log_without_SAR.csv')
    print('Data loaded.')
    #banks = set(df['bankOrig'].unique().tolist() + df['bankDest'].unique().tolist())
    banks=['bank']
    #banks = ['handelsbanken', 'swedbank']
    overlap = 0.5 # overlap of training and testing data
    print('start')
    for bank in banks:
        print('For bank', bank)
        df_bank = df[(df['bankOrig'] == bank) | (df['bankDest'] == bank)]
        train_start = df_bank['step'].min()
        train_end = df_bank['step'].min() + (df_bank['step'].max() - df_bank['step'].min()) * (overlap+(1-overlap)/2)
        test_start = df_bank['step'].min() + (df_bank['step'].max() - df_bank['step'].min()) * (1-overlap)/2
        test_end = df_bank['step'].max()
        
        df_bank_train = df_bank[(df_bank['step'] >= train_start) & (df_bank['step'] <= train_end)]
        df_bank_test = df_bank[(df_bank['step'] >= test_start) & (df_bank['step'] <= test_end)]
        
        print('get train nodes for bank: ', bank)
        df_nodes_train = get_nodes(df_bank_train, bank)
        df_edges_train = get_edges(df_bank_train[(df_bank_train['bankOrig'] == bank) & (df_bank_train['bankDest'] == bank)], aggregated=True, directional=False) # TODO: enable edges to/from the bank? the node features use these txs but unclear how to ceate a edge in this case, the edge can't be connected to a node with node features (could create node features based on edge txs, then the node features and edge features will look the same and some node features will be missing)
        print('get test nodes for bank: ', bank)
        df_nodes_test = get_nodes(df_bank_test, bank)
        df_edges_test = get_edges(df_bank_test[(df_bank_test['bankOrig'] == bank) & (df_bank_test['bankDest'] == bank)], aggregated=True, directional=False)
        
        df_nodes_train.reset_index(inplace=True)
        node_to_index = pd.Series(df_nodes_train.index, index=df_nodes_train['account']).to_dict()
        df_edges_train['src'] = df_edges_train['src'].map(node_to_index) # OBS: in the csv files it looks like the edge src refers to the node two rows above the acculat node, this is due to the column head and that it starts counting at 0
        df_edges_train['dst'] = df_edges_train['dst'].map(node_to_index)
        #df_nodes_train.drop(columns=['account'], inplace=True)
        
        df_nodes_test.reset_index(inplace=True)
        node_to_index = pd.Series(df_nodes_test.index, index=df_nodes_test['account']).to_dict()
        df_edges_test['src'] = df_edges_test['src'].map(node_to_index)
        df_edges_test['dst'] = df_edges_test['dst'].map(node_to_index)
        #df_nodes_test.drop(columns=['account'], inplace=True)

        os.makedirs(f'data/{DATASET}/{bank}/train', exist_ok=True)
        os.makedirs(f'data/{DATASET}/{bank}/test', exist_ok=True)
        
        df_nodes_train.to_csv(f'data/{DATASET}/{bank}/train/nodes.csv', index=False)
        df_edges_train.to_csv(f'data/{DATASET}/{bank}/train/edges.csv', index=False)
        df_nodes_test.to_csv(f'data/{DATASET}/{bank}/test/nodes.csv', index=False)
        df_edges_test.to_csv(f'data/{DATASET}/{bank}/test/edges.csv', index=False)
    
    t = time.time() - t
    print(f'Preprocessing finished in {t:.4f} seconds.')

if __name__ == "__main__":
    main()