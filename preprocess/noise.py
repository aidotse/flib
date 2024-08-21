import pandas as pd
from typing import Optional, Union, List


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


def topology_noise(nodes:pd.DataFrame, alert_members:pd.DataFrame, topologies:list=['fan_in', 'fan_out', 'stack'], fracs:Optional[Union[float, List[float]]]=None, seed:int=42):
    
    # TODO: remove alert_members, future updates should ensure that nodes has the necessary information to find the topologies
    
    if 'true_label' not in nodes.columns:
        nodes['true_label'] = nodes['is_sar']
    
    if isinstance(fracs, list):
        assert len(fracs) == len(topologies), 'fracs should have the same length as topologies'
    elif isinstance(fracs, float):
        fracs = [fracs] * len(topologies)
    elif fracs is None:
        fracs = [0.5] * len(topologies)
    
    node_sar = nodes[nodes['is_sar'] == 1][['account','is_sar']]
    alert_members = alert_members[['reason','accountID']].rename(columns={'accountID':'account'})
    alert_members = alert_members[alert_members['account'].isin(node_sar['account'])]
    
    for topology, frac in zip(topologies, fracs):
        accounts_to_flip = alert_members[alert_members['reason'] == topology]['account'].sample(frac=frac, random_state=seed)
        nodes.loc[nodes['account'].isin(accounts_to_flip), 'is_sar'] = 0
    
    return nodes


def main():
    pass

if __name__ == '__main__':
    main()