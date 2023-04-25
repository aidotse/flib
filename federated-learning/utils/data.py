import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer

class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X, 
        self.y = y
    
    def __getitem__(self, idx):
        X = self.X[0][idx].astype('float32') 
        y = self.y[idx].astype('float32')
        return X, y
    
    def __len__(self):
        return len(self.y)

class AdultDataset(Dataset):
    def __init__(self, df, target_name:str='income'):
        if type(df) == str:
            df = pd.read_csv(df)
        df = df.drop(columns=['workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'race'])
        df.loc[df['sex'] == ' Male', 'sex'] = 1
        df.loc[df['sex'] == ' Female', 'sex'] = 0
        df.loc[df['income'] == ' >50K', 'income'] = 1
        df.loc[df['income'] == ' <=50K', 'income'] = 0
        feture_names = df.columns.drop(target_name)
        ct = make_column_transformer(
            #(MinMaxScaler(), feture_names),
            (StandardScaler(), feture_names),
            remainder='passthrough'
        )
        self.features = ct.fit_transform(df[feture_names]).astype(dtype='float32')
        #self.features = df[feture_names].to_numpy(dtype='float32')
        self.targets = df[target_name].to_numpy(dtype='float32')
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx:int):
        X = self.features[idx]
        y = self.targets[idx]
        return X, y
    
    def get_targets(self):
        return self.targets

def latent_dirchlet_allocation(df:pd.DataFrame, target:str, partition:str, n:int, alpha:float):
    if partition == "homo":
        df = df.sample(frac=1)
        dfs = np.array_split(df, n)
    elif partition == "hetero":
        n_sampels, _ = df.shape
        unique_targets = df[target].unique()
        min_size = 0
        while min_size < 10:
            idx_subsets = [[] for _ in range(n)]
            for unique_target in unique_targets:
                idxs_target = df.index[df[target] == unique_target].to_numpy()
                np.random.shuffle(idxs_target)
                proportions = np.random.dirichlet(np.repeat(alpha, n))
                proportions = np.array(
                    [
                        p * (len(idx_j) < n_sampels / n)
                        for p, idx_j in zip(proportions, idx_subsets)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idxs_target)).astype(int)[:-1]
                idx_subsets = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_subsets, np.split(idxs_target, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_subsets])
        dfs = []
        for idx_subset in idx_subsets:
            np.random.shuffle(idx_subset)
            df_subset = df.iloc[idx_subset]
            df_subset = df_subset.reset_index(drop=True)
            dfs.append(df_subset)
        
    return dfs

def label_counts(df:pd.DataFrame, target:str):
    return df[target].value_counts().to_dict()

'''
df = pd.read_csv('data/adult/adult.csv')
data = AdultDataset(df=df)
x, y = data.__getitem__(0)
print(x)
print(y)
'''