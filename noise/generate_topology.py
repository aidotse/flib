import pandas as pd
import os


datasets = ['100K_accts_EASY25_NEW_NEW','100K_accts_MID5_NEW_NEW','100K_accts_HARD1_NEW_NEW']
extra_path = '/bank/train/'
ratio = 'no_noise'
noise = 'nodes_missing_label.csv'

for dataset in datasets:
    data_path = dataset + extra_path + ratio + '/' + noise
    data = pd.read_csv(data_path)

    alert_members_path = dataset + extra_path + 'alert_members.csv'
    alert_members = pd.read_csv(alert_members_path)
    alert_members = alert_members[['reason','accountID']]
    # Rename accountID to account
    alert_members = alert_members.rename(columns={'accountID':'account'})

    # Make a dataframe from data where is_sar = 1
    data_sar = data[data['is_sar'] == 1]
    data_sar = data_sar[['account','is_sar']]

    print(alert_members.shape)

    # In alert_members, remove the row if the account is not in data_sar
    alert_members = alert_members[alert_members['account'].isin(data_sar['account'])]

    print(alert_members.shape)

    # Create a list of the unique reasons
    reasons = alert_members['reason'].unique()

    for reason in reasons:
        print(reason)
        # Create a list of the accounts that have the reason
        accounts_to_flip = alert_members[alert_members['reason'] == reason]['account']
        # Sample 10% of the accounts
        accounts_to_flip_10 = accounts_to_flip.sample(frac=0.1, random_state=42)
        accounts_to_flip_25 = accounts_to_flip.sample(frac=0.25, random_state=42)
        accounts_to_flip_100 = accounts_to_flip.sample(frac=1, random_state=42)

        data10 = data.copy()
        data25 = data.copy()
        data100 = data.copy()

        # In data, flip the is_sar value for the accounts in accounts_to_flip
        data10.loc[data10['account'].isin(accounts_to_flip_10), 'is_sar'] = 0
        data25.loc[data25['account'].isin(accounts_to_flip_25), 'is_sar'] = 0
        data100.loc[data100['account'].isin(accounts_to_flip_100), 'is_sar'] = 0

        # Save the new data
        write_path10 = dataset + extra_path + '0.1' + '/' + f'nodes_{reason}.csv'
        write_path25 = dataset + extra_path + '0.25' + '/' + f'nodes_{reason}.csv'
        write_path100 = dataset + extra_path + '1.0' + '/' + f'nodes_{reason}.csv'
        data10.to_csv(write_path10, index=False)
        data25.to_csv(write_path25, index=False)
        data100.to_csv(write_path100, index=False)