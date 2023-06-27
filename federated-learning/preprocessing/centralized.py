import pandas as pd
import numpy as np
import os

def num_incoming_txs(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    num = df[df['nameDest'] == account].shape[0]
    return num

def sum_incoming_txs(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    sum = df[df['nameDest'] == account]['amount'].sum()
    return sum

def freq_incoming_txs(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    freq = df[df['nameDest'] == account].shape[0] / (range[1] - range[0])
    return freq

def num_outgoing_txs(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    num = df[df['nameOrig'] == account].shape[0]
    return num

def sum_outgoing_txs(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    sum = df[df['nameOrig'] == account]['amount'].sum()
    return sum

def freq_outgoing_txs(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    freq = df[df['nameOrig'] == account].shape[0] / (range[1] - range[0])
    return freq

def num_txs(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    num = df[(df['nameOrig'] == account) | (df['nameDest'] == account)].shape[0]
    return num

def sum_txs(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    sum = df[(df['nameOrig'] == account) | (df['nameDest'] == account)]['amount'].sum()
    return sum

def freq_txs(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    freq = df[(df['nameOrig'] == account) | (df['nameDest'] == account)].shape[0] / (range[1] - range[0])
    return freq

def num_unique_counterparties(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    unique_incoming = df[df['nameDest'] == account]['nameOrig'].unique()
    unique_outgoing = df[df['nameOrig'] == account]['nameDest'].unique()
    unique = np.unique(np.concatenate((unique_incoming, unique_outgoing), axis=0))
    num = unique.shape[0]
    return num

def freq_unique_counterparties(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    unique_incoming = df[df['nameDest'] == account]['nameOrig'].unique()
    unique_outgoing = df[df['nameOrig'] == account]['nameDest'].unique()
    unique = np.unique(np.concatenate((unique_incoming, unique_outgoing), axis=0))
    freq = unique.shape[0] / (range[1] - range[0])
    return freq

def num_phone_changes(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    df = df[(df['nameOrig'] == account) | (df['nameDest'] == account)]
    if df.empty:
        return 0
    if df.iloc[0]['nameOrig'] == account:
        num_start = df.iloc[0]['phoneChangesOrig']
    else:
        num_start = df.iloc[0]['phoneChangesDest']
    if df.iloc[-1]['nameOrig'] == account:
        num_end = df.iloc[-1]['phoneChangesOrig']
    else:
        num_end = df.iloc[-1]['phoneChangesDest']
    num = num_end - num_start
    return num

def freq_phone_changes(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    df = df[(df['nameOrig'] == account) | (df['nameDest'] == account)]
    if df.empty:
        return 0
    if df.iloc[0]['nameOrig'] == account:
        num_start = df.iloc[0]['phoneChangesOrig']
    else:
        num_start = df.iloc[0]['phoneChangesDest']
    if df.iloc[-1]['nameOrig'] == account:
        num_end = df.iloc[-1]['phoneChangesOrig']
    else:
        num_end = df.iloc[-1]['phoneChangesDest']
    num = num_end - num_start
    freq = num / (range[1] - range[0])
    return freq

def num_bank_changes(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    bankOrigs = df[df['nameOrig'] == account]['bankOrig']
    bankDests = df[df['nameDest'] == account]['bankDest']
    unique_banks = np.unique(np.concatenate((bankOrigs, bankDests), axis=0))
    num = unique_banks.shape[0] - 1
    return num

def freq_bank_changes(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    bankOrigs = df[df['nameOrig'] == account]['bankOrig']
    bankDests = df[df['nameDest'] == account]['bankDest']
    unique_banks = np.unique(np.concatenate((bankOrigs, bankDests), axis=0))
    num = unique_banks.shape[0] - 1
    freq = num / (range[1] - range[0])
    freq = 0 if freq < 0 else freq
    return freq

def is_sar(df, account):
    if df[df['nameOrig'] == account]['isSAR'].sum() > 0:
        is_sar = 1
    elif df[df['nameDest'] == account]['isSAR'].sum() > 0:
        is_sar = 1
    else:
        is_sar = 0
    return is_sar


def main():
    DATASET = '10K_accts'
    df = pd.read_csv(f'/home/edvin/Desktop/flib/AMLsim/outputs/{DATASET}/tx_log.csv')
    min_step = df['step'].min()
    max_step = df['step'].max()
    ranges = [[min_step, max_step], [max_step // 2, max_step], [2 * max_step // 3, max_step]]
    nameOrigs = df['nameOrig'].unique()
    nameDests = df['nameDest'].unique()
    accounts = np.unique(np.concatenate((nameOrigs, nameDests), axis=0))
    processed_df = pd.DataFrame(columns=[
        'account', 
        'num_outgoing_txs_1', 'num_outgoing_txs_2', 'num_outgoing_txs_3',
        'sum_outgoing_txs_1', 'sum_outgoing_txs_2', 'sum_outgoing_txs_3',
        'freq_outgoing_txs_1', 'freq_outgoing_txs_2', 'freq_outgoing_txs_3',
        'num_txs_1', 'num_txs_2', 'num_txs_3',
        'sum_txs_1', 'sum_txs_2', 'sum_txs_3',
        'freq_txs_1', 'freq_txs_2', 'freq_txs_3',
        'num_unique_counterparties_1', 'num_unique_counterparties_2', 'num_unique_counterparties_3',
        'freq_unique_counterparties_1', 'freq_unique_counterparties_2', 'freq_unique_counterparties_3',
        'num_phone_changes_1', 'num_phone_changes_2', 'num_phone_changes_3',
        'freq_phone_changes_1', 'freq_phone_changes_2', 'freq_phone_changes_3',
        'num_bank_changes_1', 'num_bank_changes_2', 'num_bank_changes_3',
        'freq_bank_changes_1', 'freq_bank_changes_2', 'freq_bank_changes_3',
        'is_sar'
    ])
    for account in accounts:
        processed_df = processed_df.append({
            'account': account, 
            'num_outgoing_txs_1': num_txs(df, account, ranges[0]), 'num_outgoing_txs_2': num_txs(df, account, ranges[1]), 'num_outgoing_txs_3': num_txs(df, account, ranges[2]),
            'sum_outgoing_txs_1': sum_txs(df, account, ranges[0]), 'sum_outgoing_txs_2': sum_txs(df, account, ranges[1]), 'sum_outgoing_txs_3': sum_txs(df, account, ranges[2]),
            'freq_outgoing_txs_1': freq_txs(df, account, ranges[0]), 'freq_outgoing_txs_2': freq_txs(df, account, ranges[1]), 'freq_outgoing_txs_3': freq_txs(df, account, ranges[2]),
            'num_txs_1': num_txs(df, account, ranges[0]), 'num_txs_2': num_txs(df, account, ranges[1]), 'num_txs_3': num_txs(df, account, ranges[2]),
            'sum_txs_1': sum_txs(df, account, ranges[0]), 'sum_txs_2': sum_txs(df, account, ranges[1]), 'sum_txs_3': sum_txs(df, account, ranges[2]),
            'freq_txs_1': freq_txs(df, account, ranges[0]), 'freq_txs_2': freq_txs(df, account, ranges[1]), 'freq_txs_3': freq_txs(df, account, ranges[2]),
            'num_unique_counterparties_1': num_unique_counterparties(df, account, ranges[0]), 'num_unique_counterparties_2': num_unique_counterparties(df, account, ranges[1]), 'num_unique_counterparties_3': num_unique_counterparties(df, account, ranges[2]),
            'freq_unique_counterparties_1': freq_unique_counterparties(df, account, ranges[0]), 'freq_unique_counterparties_2': freq_unique_counterparties(df, account, ranges[1]), 'freq_unique_counterparties_3': freq_unique_counterparties(df, account, ranges[2]),
            'num_phone_changes_1': num_phone_changes(df, account, ranges[0]), 'num_phone_changes_2': num_phone_changes(df, account, ranges[1]), 'num_phone_changes_3': num_phone_changes(df, account, ranges[2]),
            'freq_phone_changes_1': freq_phone_changes(df, account, ranges[0]), 'freq_phone_changes_2': freq_phone_changes(df, account, ranges[1]), 'freq_phone_changes_3': freq_phone_changes(df, account, ranges[2]),
            'num_bank_changes_1': num_bank_changes(df, account, ranges[0]), 'num_bank_changes_2': num_bank_changes(df, account, ranges[1]), 'num_bank_changes_3': num_bank_changes(df, account, ranges[2]),
            'freq_bank_changes_1': freq_bank_changes(df, account, ranges[0]), 'freq_bank_changes_2': freq_bank_changes(df, account, ranges[1]), 'freq_bank_changes_3': freq_bank_changes(df, account, ranges[2]),
            'is_sar': is_sar(df, account)
        }, ignore_index=True)
    os.makedirs(f'/home/edvin/Desktop/flib/federated-learning/datasets/{DATASET}', exist_ok=True)
    processed_df.to_csv(f'/home/edvin/Desktop/flib/federated-learning/datasets/{DATASET}/data.csv', index=False)
    
if __name__ == '__main__':
    main()
