import pandas as pd

def flip_labels(nodes:pd.DataFrame, labels:list=[0, 1], fracs:list=[0.01, 0.1], seed:int=42):
    
    if 'true_label' not in nodes.columns:
        nodes['true_label'] = nodes['is_sar']
    
    for label, frac in zip(labels, fracs):
        accounts_to_flip = nodes[nodes['true_label'] == label]['account'].sample(frac=frac, random_state=seed)
        nodes.loc[nodes['account'].isin(accounts_to_flip), 'is_sar'] = 1 - label
    
    return nodes


def missing_labels(nodes:pd.DataFrame, labels:list=[0, 1], fracs:list=[0.01, 0.1], seed:int=42):
    
    if 'true_label' not in nodes.columns:
        nodes['true_label'] = nodes['is_sar']
    
    for label, frac in zip(labels, fracs):
        accounts_to_miss = nodes[nodes['true_label'] == label]['account'].sample(frac=frac, random_state=seed)
        nodes.loc[nodes['account'].isin(accounts_to_miss), 'is_sar'] = -1
    
    return nodes


def flip_neighbours(nodes:pd.DataFrame, edges:pd.DataFrame, frac:float=0.1, seed:int=42):
    
    if 'true_label' not in nodes.columns:
        nodes['true_label'] = nodes['is_sar']
    
    # find all normal accounts that are connected to SAR accounts
    sar_accounts = set(nodes[nodes['true_label'] == 1]['account'])
    edges_with_sar = edges[edges['src'].isin(sar_accounts) | edges['dst'].isin(sar_accounts)]
    normal_accounts_with_edges_to_sar_accounts = set(edges_with_sar['src']).union(set(edges_with_sar['dst'])) - sar_accounts
    normal_accounts_with_edges_to_sar_accounts = normal_accounts_with_edges_to_sar_accounts.intersection(set(nodes[nodes['true_label'] == 0]['account']))
    normal_accounts_with_edges_to_sar_accounts = pd.DataFrame(normal_accounts_with_edges_to_sar_accounts, columns=['account'])
    normal_accounts_with_edges_to_sar_accounts = normal_accounts_with_edges_to_sar_accounts.sort_values(by='account')
    
    # flip labels of normal accounts that are connected to SAR accounts
    accounts_to_flip = normal_accounts_with_edges_to_sar_accounts['account'].sample(frac=frac, random_state=seed)
    nodes.loc[nodes['account'].isin(accounts_to_flip), 'is_sar'] = 1 - nodes.loc[nodes['account'].isin(accounts_to_flip), 'true_label']
    
    return nodes, edges


def topology_noise():
    return


def main():
    
    # create dummy data
    nodes = pd.DataFrame({
        'account': ['a100', 'a101', 'a102', 'a103', 'a104', 'a105', 'a106', 'a107', 'a108', 'a109'], 
        'is_sar':[0, 0, 1, 0, 0, 0, 1, 1, 0, 1]
    })
    
    # test functions
    
    # nodes_flipped_labels = flip_labels(nodes=nodes, labels=[0, 1], fracs=[0.17, 0.25], seed=42)
    # print(nodes_flipped_labels)
    
    # nodes_missing_labels = missing_labels(nodes=nodes, labels=[0, 1], fracs=[0.17, 0.25], seed=42)
    # print(nodes_missing_labels)
    
    # create dummy data
    nodes = pd.DataFrame({
        'account': ['a100', 'a101', 'a102', 'a103', 'a104'], 
        'is_sar':[1, 0, 0, 0, 0]
    })
    edges = pd.DataFrame({
        'src': ['a100', 'a100', 'a100', 'a100'], 
        'dst': ['a101', 'a102', 'a103', 'a104'],
    })
    
    nodes_flipped_neighbours, edges = flip_neighbours(nodes=nodes, edges=edges, frac=0.5, seed=42)
    print(nodes_flipped_neighbours)

    

if __name__ == '__main__':
    main()