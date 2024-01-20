import pandas as pd
import numpy as np
import os
import multiprocessing as mp

def load_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    df.drop(columns=['type', 'daysInBankOrig', 'daysInBankDest', 'oldbalanceOrig', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'phoneChangesOrig', 'phoneChangesDest', 'alertID', 'modelType'], inplace=True)
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

def get_nodes(df:pd.DataFrame) -> pd.DataFrame:
    nodes = cal_node_features(df)
    return nodes

def get_edges(df:pd.DataFrame, aggregated:bool=True, directional:bool=False) -> pd.DataFrame:
    if aggregated:
        edges = cal_edge_features(df, directional)
    elif not aggregated:
        edges = df[['step', 'nameOrig', 'nameDest', 'amount', 'isSAR']].rename(columns={'step': 't', 'nameOrig': 'src', 'nameDest': 'dst', 'isSAR': 'is_sar'})
    return edges
    
def cal_node_features(df:pd.DataFrame) -> pd.DataFrame:
    df1 = df[['nameOrig', 'amount', 'nameDest', 'isSAR']].rename(columns={'nameOrig': 'account', 'nameDest': 'counterpart', 'isSAR': 'is_sar'})
    df2 = df[['nameDest', 'amount', 'nameOrig', 'isSAR']].rename(columns={'nameDest': 'account', 'nameOrig': 'counterpart', 'isSAR': 'is_sar'})
    df2['amount'] = df2['amount'] * -1
    df = pd.concat([df1, df2])
    gb = df.groupby(['account'])
    sums = gb['amount'].sum().rename('sum')
    means = gb['amount'].mean().rename('mean')
    medians = gb['amount'].median().rename('median')
    stds = gb['amount'].std().fillna(0.0).fillna(0.0).rename('std') 
    maxs = gb['amount'].max().rename('max')
    mins = gb['amount'].min().rename('min')
    in_degrees = gb['amount'].apply(lambda x: (x>0).sum()).rename('in_degree')
    out_degrees = gb['amount'].apply(lambda x: (x<0).sum()).rename('out_degree')
    n_unique_in = gb.apply(lambda x: x[x['amount']>0]['counterpart'].nunique()).rename('n_unique_in')
    n_unique_out = gb.apply(lambda x: x[x['amount']<0]['counterpart'].nunique()).rename('n_unique_out')
    is_sar = gb['is_sar'].max().rename('is_sar')
    df = pd.concat([sums, means, medians, stds, maxs, mins, in_degrees, out_degrees, n_unique_in, n_unique_out, is_sar], axis=1)
    return df

def cal_edge_features(df:pd.DataFrame, directional:bool=False) -> pd.DataFrame:
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

def main():
    DATASET = '100K_accts'
    path = f'../AMLsim/outputs/{DATASET}/tx_log.csv'
    df = load_data(path)
    banks = set(df['bankOrig'].unique().tolist() + df['bankDest'].unique().tolist())
    test_size = 0.2
    
    for bank in banks:
        df_bank = df[(df['bankOrig'] == bank) | (df['bankDest'] == bank)]
        split_step = (df_bank['step'].max() - df_bank['step'].min()) * (1 - test_size) + df_bank['step'].min()
        
        df_bank_train = df_bank[df_bank['step'] <= split_step]
        df_bank_test = df_bank #[df_bank['step'] > split_step]
        
        df_nodes_train = get_nodes(df_bank_train)
        df_edges_train = get_edges(df_bank_train, aggregated=True, directional=False)
        df_nodes_test = get_nodes(df_bank_test)
        df_edges_test = get_edges(df_bank_test, aggregated=True, directional=False)
        
        df_nodes_test.reset_index(inplace=True)
        node_to_index = pd.Series(df_nodes_test.index, index=df_nodes_test['account']).to_dict()
        df_edges_test['src'] = df_edges_test['src'].map(node_to_index)
        df_edges_test['dst'] = df_edges_test['dst'].map(node_to_index)
        df_nodes_test.drop(columns=['account'], inplace=True)
        
        os.makedirs(f'data/{DATASET}/{bank}/train', exist_ok=True)
        os.makedirs(f'data/{DATASET}/{bank}/test', exist_ok=True)
        
        df_nodes_train.to_csv(f'data/{DATASET}/{bank}/train/nodes.csv', index=False)
        df_edges_train.to_csv(f'data/{DATASET}/{bank}/train/edges.csv', index=False)
        df_nodes_test.to_csv(f'data/{DATASET}/{bank}/test/nodes.csv', index=False)
        df_edges_test.to_csv(f'data/{DATASET}/{bank}/test/edges.csv', index=False)

if __name__ == "__main__":
    main()