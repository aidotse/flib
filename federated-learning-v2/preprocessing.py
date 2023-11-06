import pandas as pd
import multiprocessing as mp
import time
import scipy as sp
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore')

def load_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(columns=['type', 'daysInBankOrig', 'daysInBankDest', 'oldbalanceOrig', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'phoneChangesOrig', 'phoneChangesDest', 'alertID', 'modelType'], inplace=True)
    return df

def cal_stats(df:pd.DataFrame, range:list=None, direction:str='both', include_source_sink:bool=False) -> pd.DataFrame:
    if not include_source_sink:
        df = df[(df['account'] != -1) & (df['account'] != -2)]
    if range:
        df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    if direction == 'incoming':
        df = df.loc[df['amount'] > 0.0]
    elif direction == 'outgoing':
        df = df.loc[df['amount'] < 0.0]
        df['amount'] = df['amount'].abs()    
    gb = df.groupby(['account'])
    sums = gb['amount'].sum()
    means = gb['amount'].mean()
    medians = gb['amount'].median()
    stds = gb['amount'].std()
    maxs = gb['amount'].max()
    mins = gb['amount'].min()
    degrees = gb['counterpart'].count()
    uniques = gb['counterpart'].nunique()
    df = pd.concat([sums, means, medians, stds, maxs, mins, degrees, uniques], axis=1)
    suffix = ''
    if range:
        suffix += f'_{range[0]}_{range[1]}'
    df.columns = ['sum'+suffix, 'mean'+suffix, 'median'+suffix, 'std'+suffix, 'max'+suffix, 'min'+suffix, 'degree'+suffix, 'unique'+suffix]
    return df

def cal_label(df:pd.DataFrame) -> pd.DataFrame:
    gb = df.groupby(['account'])
    is_sar = gb['is_sar'].max().to_frame()
    return is_sar

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
    
def split_and_reform(df:pd.DataFrame, bank:str) -> pd.DataFrame:
    df1 = df[df['bankOrig'] == bank]
    df1 = df1.drop(columns=['bankOrig', 'bankDest'])
    df1.rename(columns={'nameOrig': 'account', 'nameDest': 'counterpart', 'isSAR': 'is_sar'}, inplace=True)
    
    df2 = df[df['bankDest'] == bank]
    df2 = df2.drop(columns=['bankOrig', 'bankDest'])
    df2.rename(columns={'nameDest': 'account', 'nameOrig': 'counterpart', 'isSAR': 'is_sar'}, inplace=True)
    
    df2['amount'] = df2['amount'] * -1
    return pd.concat([df1, df2])

def main():
    DATASET = '50K_accts'
    path = f'../AMLsim/outputs/{DATASET}/tx_log.csv'
    os.makedirs(f'../datasets/{DATASET}/preprocessed', exist_ok=True)
    df = load_data(path)
    banks = set(df['bankOrig'].unique().tolist() + df['bankDest'].unique().tolist())
    banks.remove('sink')
    banks.remove('source')
    ranges = [[0, int(365/4)], [int(365/4), int(365/2)], [int(365/2), int(365/4*3)], [int(365/4*3), 365]]
    for bank in banks:
        df_bank = split_and_reform(df, bank)
        df_stats1 = cal_stats(df_bank, ranges[0])
        df_stats2 = cal_stats(df_bank, ranges[1])
        df_stats3 = cal_stats(df_bank, ranges[2])
        df_stats4 = cal_stats(df_bank, ranges[3])
        df_spending_behavior = cal_spending_behavior(df_bank)
        df_label = cal_label(df_bank)
        df_features = pd.merge(df_stats1, df_stats2, on='account')
        df_features = pd.merge(df_features, df_stats3, on='account')
        df_features = pd.merge(df_features, df_stats4, on='account')
        df_features = pd.merge(df_features, df_spending_behavior, on='account')
        df_features = pd.merge(df_features, df_label, on='account')
        df_features.to_csv(f'../datasets/{DATASET}/preprocessed/{bank}.csv', index=False)
        print(bank + ' done')

if __name__ == '__main__':
    main()