import pandas as pd
import os

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

def get_edges(df:pd.DataFrame) -> pd.DataFrame:
    edges = df[['nameOrig', 'nameDest']].rename(columns={'nameOrig': 'src', 'nameDest': 'dst'})
    edges.drop_duplicates(inplace=True)
    return edges
    
def cal_node_features(df:pd.DataFrame) -> pd.DataFrame:
    df1 = df[['nameOrig', 'amount', 'nameDest', 'isSAR']].rename(columns={'nameOrig': 'account', 'nameDest': 'counterpart', 'isSAR': 'is_sar'})
    df2 = df[['nameDest', 'amount', 'nameOrig', 'isSAR']].rename(columns={'nameDest': 'account', 'nameOrig': 'counterpart', 'isSAR': 'is_sar'})
    df2['amount'] = df2['amount'] * -1
    df = pd.concat([df1, df2])
    gb = df.groupby(['account'])
    sums = gb['amount'].sum()
    means = gb['amount'].mean()
    medians = gb['amount'].median()
    stds = gb['amount'].std().fillna(0.0)
    maxs = gb['amount'].max()
    mins = gb['amount'].min()
    in_degrees = gb['amount'].apply(lambda x: (x>0).sum())
    out_degrees = gb['amount'].apply(lambda x: (x<0).sum())
    unique_in_degrees = gb.apply(lambda x: x[x['amount']>0]['counterpart'].nunique())
    unique_out_degrees = gb.apply(lambda x: x[x['amount']<0]['counterpart'].nunique())
    y = gb['is_sar'].max()
    df = pd.concat([sums, means, medians, stds, maxs, mins, in_degrees, out_degrees, unique_in_degrees, unique_out_degrees, y], axis=1)
    df.columns = [f'x{i}' for i in range(1, 11)] + ['y']
    return df

def cal_label(df:pd.DataFrame) -> pd.DataFrame:
    gb = df.groupby(['account'])
    is_sar = gb['is_sar'].max().to_frame()
    return is_sar

def main():
    DATASET = '1bank'
    path = f'../AMLsim/outputs/{DATASET}/tx_log.csv'
    df = load_data(path)
    banks = set(df['bankOrig'].unique().tolist() + df['bankDest'].unique().tolist())
    test_size = 0.2
    
    for bank in banks:
        df_bank = df[(df['bankOrig'] == bank) | (df['bankDest'] == bank)]
        split_step = (df_bank['step'].max() - df_bank['step'].min()) * (1 - test_size) + df_bank['step'].min()
        
        df_bank_train = df_bank[df_bank['step'] <= split_step]
        df_bank_test = df_bank[df_bank['step'] > split_step]
        
        df_nodes_train = get_nodes(df_bank_train)
        df_edges_train = get_edges(df_bank_train)
        df_nodes_test = get_nodes(df_bank_test)
        df_edges_test = get_edges(df_bank_test)
        
        df_nodes_test.reset_index(inplace=True)
        node_to_index = pd.Series(df_nodes_test.index, index=df_nodes_test['account']).to_dict()
        df_edges_test['src'] = df_edges_test['src'].map(node_to_index)
        df_edges_test['dst'] = df_edges_test['dst'].map(node_to_index)
        
        os.makedirs(f'data/{DATASET}/{bank}/trainset', exist_ok=True)
        os.makedirs(f'data/{DATASET}/{bank}/testset', exist_ok=True)
        
        df_nodes_train.to_csv(f'data/{DATASET}/{bank}/trainset/nodes.csv')
        df_edges_train.to_csv(f'data/{DATASET}/{bank}/trainset/edges.csv', index=False)
        df_nodes_test.to_csv(f'data/{DATASET}/{bank}/testset/nodes.csv')
        df_edges_test.to_csv(f'data/{DATASET}/{bank}/testset/edges.csv', index=False)

if __name__ == "__main__":
    main()