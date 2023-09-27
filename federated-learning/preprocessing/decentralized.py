import pandas as pd
import numpy as np
import os
from multiprocessing import Pool

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

def num_days_in_bank(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    daysInBankOrigs = df[df['nameOrig'] == account]['daysInBankOrig'].max()
    daysInBankDests = df[df['nameDest'] == account]['daysInBankDest'].max()
    num = max(0, daysInBankOrigs, daysInBankDests)
    return num

def freq_days_in_bank(df, account, range):
    df = df[(df['step'] > range[0]) & (df['step'] < range[1])]
    daysInBankOrigs = df[df['nameOrig'] == account]['daysInBankOrig'].max()
    daysInBankDests = df[df['nameDest'] == account]['daysInBankDest'].max()
    num = max(0, daysInBankOrigs, daysInBankDests)
    freq = num / (range[1] - range[0])
    return freq

def is_sar(df, account):
    if df[df['nameOrig'] == account]['isSAR'].sum() > 0:
        is_sar = 1
    elif df[df['nameDest'] == account]['isSAR'].sum() > 0:
        is_sar = 1
    else:
        is_sar = 0
    return is_sar

def process_accounts(args):
    df, accounts, ranges = args
    account_features = []
    for account in accounts:
        account_features.append({
            'account': account, 
            'num_incomeing_txs_1': num_incoming_txs(df, account, ranges[0]), 'num_incomeing_txs_2': num_incoming_txs(df, account, ranges[1]), 'num_incomeing_txs_3': num_incoming_txs(df, account, ranges[2]),
            'sum_incomeing_txs_1': sum_incoming_txs(df, account, ranges[0]), 'sum_incomeing_txs_2': sum_incoming_txs(df, account, ranges[1]), 'sum_incomeing_txs_3': sum_incoming_txs(df, account, ranges[2]),
            'freq_incomeing_txs_1': freq_incoming_txs(df, account, ranges[0]), 'freq_incomeing_txs_2': freq_incoming_txs(df, account, ranges[1]), 'freq_incomeing_txs_3': freq_incoming_txs(df, account, ranges[2]),
            'num_outgoing_txs_1': num_outgoing_txs(df, account, ranges[0]), 'num_outgoing_txs_2': num_outgoing_txs(df, account, ranges[1]), 'num_outgoing_txs_3': num_outgoing_txs(df, account, ranges[2]),
            'sum_outgoing_txs_1': sum_outgoing_txs(df, account, ranges[0]), 'sum_outgoing_txs_2': sum_outgoing_txs(df, account, ranges[1]), 'sum_outgoing_txs_3': sum_outgoing_txs(df, account, ranges[2]),
            'freq_outgoing_txs_1': freq_outgoing_txs(df, account, ranges[0]), 'freq_outgoing_txs_2': freq_outgoing_txs(df, account, ranges[1]), 'freq_outgoing_txs_3': freq_outgoing_txs(df, account, ranges[2]),
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
        })
    return account_features

def main():
    DATASET = '10K_accts_easy'
    df = pd.read_csv(f'/home/edvin/Desktop/flib/AMLsim/outputs/{DATASET}/tx_log.csv')
    #df = df.sample(n=10000).reset_index(drop=True)
    bankOrigs = df['bankOrig'].unique()
    bankDests = df['bankDest'].unique()
    banks = np.unique(np.concatenate((bankOrigs, bankDests), axis=0))
    #banks = ['nodrea', 'seb', 'skandia', 'spabanken', 'svea', 'swedbank', 'ålandsbanken']
    
    os.makedirs(f'/home/edvin/Desktop/flib/federated-learning/datasets/{DATASET}', exist_ok=True)
    
    min_step = df['step'].min()
    max_step = df['step'].max()
    
    #accounts = df[]

    n_accts = 0
    acct_list = []
    for bank in banks:
        df_bank = df[(df['bankOrig'] == bank) | (df['bankDest'] == bank)]
        ranges = [[min_step, max_step], [max_step // 2, max_step], [2 * max_step // 3, max_step]]
        nameOrigs = df_bank[df_bank['bankOrig']==bank]['nameOrig'].unique()
        nameDests = df_bank[df_bank['bankDest']==bank]['nameDest'].unique()
        names = pd.concat([df_bank[df_bank['bankOrig']==bank]['nameOrig'], df_bank[df_bank['bankDest']==bank]['nameDest']])

        names = names.drop_duplicates().sort_values()
        acct_list.append(names)
        accounts = np.unique(np.concatenate((nameOrigs, nameDests), axis=0))
        print(f'n accts in {bank}: {len(accounts)}')
        print()
        n_accts += len(accounts)
        processed_df = pd.DataFrame(columns=[
            'account', 
            'num_incomeing_txs_1', 'num_incomeing_txs_2', 'num_incomeing_txs_3',
            'sum_incomeing_txs_1', 'sum_incomeing_txs_2', 'sum_incomeing_txs_3',
            'freq_incomeing_txs_1', 'freq_incomeing_txs_2', 'freq_incomeing_txs_3',
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
        num_workers = 8
        accounts_list = np.array_split(accounts, num_workers)
        args = [(df_bank, account, ranges) for account in accounts_list]
        with Pool(num_workers) as p:
            outputs = p.map(process_accounts, args)
        i = 0
        for output in outputs:
            for account_features in output:
                processed_df.loc[i] = account_features
                i += 1
        processed_df.to_csv(f'/home/edvin/Desktop/flib/federated-learning/datasets/{DATASET}/{bank}.csv', index=False)
    
    print(f'n accts total: {n_accts}')
    acct_list = np.unique(np.concatenate(acct_list, axis=0))
    print(f'n unique accts total: {len(acct_list)}')

if __name__ == '__main__':
    main()
