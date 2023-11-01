import pandas as pd
import multiprocessing as mp
import time
import scipy as sp
import numpy as np

def load_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(columns=['type', 'daysInBankOrig', 'daysInBankDest', 'oldbalanceOrig', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'phoneChangesOrig', 'phoneChangesDest', 'alertID', 'modelType'], inplace=True)
    return df

def cal_stats(df:pd.DataFrame, range:list=None, direction:str='both', include_source_sink:bool=False) -> pd.DataFrame:
    if not include_source_sink:
        df = df[(df['name'] != -1) & (df['name'] != -2)]
    if range:
        df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    if direction == 'incoming':
        df = df.loc[df['amount'] > 0.0]
    elif direction == 'outgoing':
        df = df.loc[df['amount'] < 0.0]
        df['amount'] = df['amount'].abs()    
    gb = df.groupby(['name'])
    sums = gb['amount'].sum()
    means = gb['amount'].mean()
    medians = gb['amount'].median()
    stds = gb['amount'].std()
    maxs = gb['amount'].max()
    mins = gb['amount'].min()
    degrees = gb['counterparty'].count()
    uniques = gb['counterparty'].nunique()
    is_sar = gb['is_sar'].max()
    df = pd.concat([sums, means, medians, stds, maxs, mins, degrees, uniques, is_sar], axis=1)
    df.columns = ['sum', 'mean', 'median', 'std', 'max', 'min', 'degree', 'unique', 'is_sar']
    return df

def anderson_ksamp_mp(samples):
    res = sp.stats.anderson_ksamp(samples)
    return res.pvalue

def cal_pvalues(df:pd.DataFrame) -> float:
    amounts = df['amount'].to_numpy()
    samples = []
    for i in range(len(amounts)):
        samples.append([amounts[i:i+1], np.delete(amounts, i)])
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pvalues = pool.map(anderson_ksamp_mp, samples)
    pvalue = np.sum(pvalues)
    return pvalue
    

def cal_spending_behavior(df:pd.DataFrame, range:list=None, interval:int=7) -> pd.DataFrame:
    if range:
        df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    df = df.loc[df['counterparty']==-2]
    df['interval_group'] = df['step'] // interval
    df['amount'] = df['amount'].abs()
    gb = df.groupby(['name', 'interval_group'])
    df = gb['amount'].mean().reset_index().drop(columns=['interval_group'])
    print(df)
    df = df.groupby(['name']).apply(cal_pvalues)
    print(df)
    
    
def split_and_reform(df:pd.DataFrame, bank:str) -> pd.DataFrame:
    df1 = df[df['bankOrig'] == bank]
    df1 = df1.drop(columns=['bankOrig', 'bankDest'])
    df1.rename(columns={'nameOrig': 'name', 'nameDest': 'counterparty', 'isSAR': 'is_sar'}, inplace=True)
    
    df2 = df[df['bankDest'] == bank]
    df2 = df2.drop(columns=['bankOrig', 'bankDest'])
    df2.rename(columns={'nameDest': 'name', 'nameOrig': 'counterparty', 'isSAR': 'is_sar'}, inplace=True)
    
    df2['amount'] = df2['amount'] * -1
    return pd.concat([df1, df2])

def main():
    
    DATASET = '100K_accts'
    path = f'../AMLsim/outputs/{DATASET}/tx_log.csv'
    df = load_data(path)
    
    
    banks = ['ålandsbanken'] # set(df['bankOrig'].unique().tolist() + df['bankDest'].unique().tolist())
    
    dfs = []
    for bank in banks:
        print(bank)
        dfs.append(split_and_reform(df, bank))
        
    t = time.time()
    print(dfs[0].columns)
    
    cal_spending_behavior(dfs[0])
    
    
    print(time.time() - t)

if __name__ == '__main__':
    main()