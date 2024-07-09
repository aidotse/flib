import pandas as pd
import os

datasets = ['100K_accts_EASY25_NEW_NEW','100K_accts_MID5_NEW_NEW','100K_accts_HARD1_NEW_NEW']
extra_path = '/bank/train/'
ratios = ['no_noise']
noise = ['nodes.csv']

for dataset in datasets:
    data_path = dataset + extra_path + ratios[0] + '/' + noise[0]

    data = pd.read_csv(data_path)

    # Create new column true_label as copy of is_sar
    data['true_label'] = data['is_sar']

    # Sample 90% of accounts with is_sar = 0
    data_normal = data[data['is_sar'] == 0].sample(frac=0.9)
    # Keep only account column
    data_normal = data_normal[['account']]


    # Sample 90% of accounts with is_sar = 1
    data_ML = data[data['is_sar'] == 1].sample(frac=0.9)
    # Keep only account column
    data_ML = data_ML[['account']]

    # For all accounts in data_normal and data_ML, set in data is_sar = -1
    data.loc[data['account'].isin(data_normal['account']), 'is_sar'] = -1
    data.loc[data['account'].isin(data_ML['account']), 'is_sar'] = -1

    # Export data to csv
    write_path = dataset + extra_path + ratios[0] + '/' + 'nodes_missing_label.csv'
    data.to_csv(write_path, index=False)