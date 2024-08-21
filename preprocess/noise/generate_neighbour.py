import pandas as pd
import os

# Define the paths
datasets = ['100K_accts_EASY25_NEW_NEW','100K_accts_MID5_NEW_NEW','100K_accts_HARD1_NEW_NEW']
extra_path = '/bank/train/'
ratio = 'no_noise'
noise = 'nodes.csv'
for dataset in datasets:
    data_path = dataset + extra_path + ratio + '/' + noise
    edges_path = dataset + extra_path + 'edges.csv'

    # Load the data
    data = pd.read_csv(data_path)
    edges = pd.read_csv(edges_path)

    # Filter data for normal and SAR accounts
    data_normal = data[data['is_sar'] == 0]
    data_normal = data_normal[['account', 'is_sar']]

    data_sar = data[data['is_sar'] == 1]
    data_sar = data_sar[['account', 'is_sar']]

    # Filter the edges data
    edges = edges[['src', 'dst']]

    # Find edges involving SAR accounts
    sar_accounts = set(data_sar['account'])
    edges_with_sar = edges[edges['src'].isin(sar_accounts) | edges['dst'].isin(sar_accounts)]

    # Find normal accounts that have an edge to or from SAR accounts
    normal_accounts_with_edges_to_sar_accounts = set(edges_with_sar['src']).union(set(edges_with_sar['dst'])) - sar_accounts
    normal_accounts_with_edges_to_sar_accounts = normal_accounts_with_edges_to_sar_accounts.intersection(set(data_normal['account']))

    # Convert the set to a DataFrame
    normal_accounts_with_edges_to_sar_accounts = pd.DataFrame(normal_accounts_with_edges_to_sar_accounts, columns=['account'])

    # Display the DataFrame
    print('Normal accounts with edges to SAR:', normal_accounts_with_edges_to_sar_accounts.shape)

    # Create a list of the unique reasons
    reason = 'neighbour'
    print(reason)

    # Create a list of the accounts that have the reason
    accounts_to_flip = normal_accounts_with_edges_to_sar_accounts

    miss_label_path = dataset + extra_path + ratio + '/' + 'nodes_missing_label.csv'
    data_miss = pd.read_csv(miss_label_path)

    data_miss_normal = data_miss[data_miss['is_sar'] == 0]
    data_miss_normasl = data_miss_normal[['account', 'is_sar']]

    # Filter accounts_to_flip to include only accounts present in data_miss_normal
    accounts_to_flip = accounts_to_flip[accounts_to_flip['account'].isin(data_miss_normal['account'])]

    print('Accounts to flip:', accounts_to_flip.shape)

    # Sample 10% and 25% of the accounts
    accounts_to_flip_10 = accounts_to_flip.sample(frac=0.1, random_state=42)
    accounts_to_flip_25 = accounts_to_flip.sample(frac=0.25, random_state=42)

    print('Accounts to flip 10%:', accounts_to_flip_10.shape)
    print('Accounts to flip 25%:', accounts_to_flip_25.shape)

    accounts_to_flip_10.to_csv(dataset + extra_path + '0.1' + '/' + f'accounts_{reason}.csv', index=False)
    
    # Create copies of the data for modification
    data10 = data_miss.copy()
    data25 = data_miss.copy()

    # In data10 and data25, flip set is_sar value to 1 for the sampled accounts
    data10.loc[data10['account'].isin(accounts_to_flip_10['account']), 'is_sar'] = 1
    data25.loc[data25['account'].isin(accounts_to_flip_25['account']), 'is_sar'] = 1

    # Verify the flipping
    print(f"Number of flipped accounts in data10: {(data10['is_sar'] == 1).sum() - (data_miss['is_sar'] == 1).sum()}")
    print(f"Number of flipped accounts in data25: {(data25['is_sar'] == 1).sum() - (data_miss['is_sar'] == 1).sum()}")

    # Save the new data
    write_path10 = dataset + extra_path + '0.1' + '/' + f'nodes_{reason}.csv'
    write_path25 = dataset + extra_path + '0.25' + '/' + f'nodes_{reason}.csv'
    data10.to_csv(write_path10, index=False)
    data25.to_csv(write_path25, index=False)

    print(f"Data saved to {write_path10} and {write_path25}")
