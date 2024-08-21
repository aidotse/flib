import pandas as pd
import os

def flip_labels(data:pd.DataFrame, labels:list=[0, 1], fracs:list=[0.01, 0.1], seed:int=42):
    
    if 'true_label' not in data.columns:
        data['true_label'] = data['is_sar']
    
    for label, frac in zip(labels, fracs):
        accounts = data[data['true_label'] == label]['account'].sample(frac=frac, random_state=seed)
        data.loc[data['account'].isin(accounts), 'is_sar'] = 1 - label
    
    return data


def missing_labels(data:pd.DataFrame, labels:list=[0, 1], fracs:list=[0.01, 0.1], seed:int=42):
    
    if 'true_label' not in data.columns:
        data['true_label'] = data['is_sar']
    
    for label, frac in zip(labels, fracs):
        accounts = data[data['true_label'] == label]['account'].sample(frac=frac, random_state=seed)
        data.loc[data['account'].isin(accounts), 'is_sar'] = -1
    
    return data


def flip_neighbours():
    return


def topology_noise():
    return


def main():
    
    # create dummy data
    data = pd.DataFrame({
        'account': ['a100', 'a101', 'a102', 'a103', 'a104', 'a105', 'a106', 'a107', 'a108', 'a109'], 
        'is_sar':[0, 0, 1, 0, 0, 0, 1, 1, 0, 1]
    })
    
    # test functions
    
    # data_flipped = flip_labels(data=data, labels=[0, 1], fracs=[0.17, 0.25], seed=42)
    # print(data_flipped)
    
    data_missing = missing_labels(data=data, labels=[0, 1], fracs=[0.17, 0.25], seed=42)
    print(data_missing)

    

if __name__ == '__main__':
    main()