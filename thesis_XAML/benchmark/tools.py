import pandas as pd 

def proccess_data(edge_data,node_data):

    bool_dict = {True: 1, False: 0}
    for key in bool_dict:
        node_data = node_data.replace(key, bool_dict[key])

    # Find unique values in the bank_id column
    bank_ids = node_data['BANK_ID'].unique()

    # Create a dictionary to map the bank_id to a number
    bank_dict = {}
    for i in range(len(bank_ids)):
        bank_dict[bank_ids[i]] = i

    for key in bank_dict:
        node_data = node_data.replace(key, bank_dict[key])
        edge_data = edge_data.replace(key, bank_dict[key])

    node_data = node_data.drop(['CUSTOMER_ID', 'ACCOUNT_TYPE','COUNTRY'], axis=1)
    #true_labels = node_data[['ACCOUNT_ID','TRUE_LABEL']]
    #true_labels = node_data[['ACCOUNT_ID','TRUE_LABEL']]
    labels = node_data[['ACCOUNT_ID','IS_SAR']]

    labels = labels.rename(columns={'ACCOUNT_ID': 'node_id'})
    #true_labels = true_labels.rename(columns={'ACCOUNT_ID': 'node_id'})
    #true_labels = true_labels.rename(columns={'ACCOUNT_ID': 'node_id'})

    init_balance_feat = node_data[['ACCOUNT_ID','INIT_BALANCE']]
    bank_id_feat = node_data[['ACCOUNT_ID','BANK_ID']]

    init_balance_feat = init_balance_feat.rename(columns={'ACCOUNT_ID': 'node_id'})
    bank_id_feat = bank_id_feat.rename(columns={'ACCOUNT_ID': 'node_id'})

    edge_data = edge_data[edge_data['type'] == 'TRANSFER']

    edge_sar = edge_data[['isSAR']]
    transactions = edge_data[['step','amount','nameOrig','bankOrig','oldbalanceOrig','newbalanceOrig','nameDest','oldbalanceDest','newbalanceDest']]
    phone_changes = edge_data[['nameOrig','phoneChangesOrig']]
    # Transactions - average amount recieved, average amount sent, total numbers of transactions, total amount recieved, total amount sent

    transaction_features = transactions.groupby('nameOrig').agg({'amount': ['mean', 'sum'], 'nameDest': 'count'}).reset_index()
    transaction_features.columns = ['nameOrig', 'avg_amount_sent', 'total_amount_sent', 'total_transactions_sent']
    transaction_features['avg_amount_sent'] = transaction_features['avg_amount_sent'].apply(lambda x: round(x, 2))
    transaction_features['total_amount_sent'] = transaction_features['total_amount_sent'].apply(lambda x: round(x, 2))

    transaction_features = transaction_features[transaction_features['nameOrig'] != -2]
    # Rename column nameOrig to node_id
    transaction_features = transaction_features.rename(columns={'nameOrig': 'node_id'})


    transaction_features_rec = transactions.groupby('nameDest').agg({'amount': ['mean', 'sum'], 'nameOrig': 'count'}).reset_index()
    transaction_features_rec.columns = ['nameDest', 'avg_amount_rec', 'total_amount_rec', 'total_transactions_rec']
    transaction_features_rec['avg_amount_rec'] = transaction_features_rec['avg_amount_rec'].apply(lambda x: round(x, 2))
    transaction_features_rec['total_amount_rec'] = transaction_features_rec['total_amount_rec'].apply(lambda x: round(x, 2))
    transaction_features_rec = transaction_features_rec[transaction_features_rec['nameDest'] != -1]
    transaction_features_rec = transaction_features_rec.rename(columns={'nameDest': 'node_id'})

    transaction_features = pd.merge(transaction_features, transaction_features_rec, on='node_id', how='outer')

    # Total number of phone changes and average number of phone changes per transaction for each node

    phone_changes_feat = phone_changes.groupby('nameOrig').agg({'phoneChangesOrig': ['mean', 'sum']}).reset_index()
    phone_changes_feat.columns = ['nameOrig', 'avg_phone_changes', 'total_phone_changes']
    phone_changes_feat['avg_phone_changes'] = phone_changes_feat['avg_phone_changes'].apply(lambda x: round(x, 2))
    phone_changes_feat = phone_changes_feat[phone_changes_feat['nameOrig'] != -2]
    phone_changes_feat = phone_changes_feat.rename(columns={'nameOrig': 'node_id'})

    # merge all features


    # Total number of unique accounts sent to and recieved from
    unique_accounts_sent = transactions.groupby('nameOrig').agg({'nameDest': 'nunique'}).reset_index()
    unique_accounts_sent = unique_accounts_sent.rename(columns={'nameOrig': 'node_id', 'nameDest': 'unique_accounts_sent'})
    unique_accounts_sent = unique_accounts_sent[unique_accounts_sent['node_id'] != -2]

    unique_accounts_rec = transactions.groupby('nameDest').agg({'nameOrig': 'nunique'}).reset_index()
    unique_accounts_rec = unique_accounts_rec.rename(columns={'nameDest': 'node_id', 'nameOrig': 'unique_accounts_rec'})
    unique_accounts_rec = unique_accounts_rec[unique_accounts_rec['node_id'] != -1]


    # sar_nodes = labels[labels['IS_SAR'] == 1]
    # # Add a feature of how many transactions recieved by SAR nodes
    # sar_nodes_rec = transactions[transactions['nameDest'].isin(sar_nodes['node_id'])]
    # sar_nodes_rec = sar_nodes_rec.groupby('nameDest').agg({'nameOrig': 'count'}).reset_index()
    # sar_nodes_rec = sar_nodes_rec.rename(columns={'nameDest': 'node_id', 'nameOrig': 'total_transactions_rec_sar'})

    # sar_nodes_sent = transactions[transactions['nameOrig'].isin(sar_nodes['node_id'])]
    # sar_nodes_sent = sar_nodes_sent.groupby('nameOrig').agg({'nameDest': 'count'}).reset_index()
    # sar_nodes_sent = sar_nodes_sent.rename(columns={'nameOrig': 'node_id', 'nameDest': 'total_transactions_sent_sar'})


    # Merge all features
    features = pd.merge(transaction_features, phone_changes_feat, on='node_id', how='outer')
    features = pd.merge(features, init_balance_feat, on='node_id', how='outer')
    features = pd.merge(features, bank_id_feat, on='node_id', how='outer')

    features = pd.merge(features, unique_accounts_sent, on='node_id', how='outer')
    features = pd.merge(features, unique_accounts_rec, on='node_id', how='outer')
    # features = pd.merge(features, sar_nodes_rec, on='node_id', how='outer')
    # features = pd.merge(features, sar_nodes_sent, on='node_id', how='outer')

    features = features.fillna(0)

    return features, labels, #true_labels

