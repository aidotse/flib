import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split
import os


# (Written by Edvin but edited)
class AmlsimDataset():
    def __init__(self, node_file:str, edge_file:str, node_features:bool=False, edge_features:bool=True, node_labels:bool=False, edge_labels:bool=False, seed:int=42):
        self.data = self.load_data(node_file, edge_file, node_features, edge_features, node_labels, edge_labels)
        
    def load_data(self, node_file, edge_file, node_features, edge_features, node_labels, edge_labels):
        nodes = pd.read_csv(node_file)
        edges = pd.read_csv(edge_file)
        edge_index = torch.tensor(edges[['src', 'dst']].values, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        
        #sort nodes according to account number from small to large
        # nodes = nodes.sort_values(by=['account'])
        print('sorting')

        nodes = nodes.drop(columns=['bank', 'account'])

        nodes = nodes.fillna(0)
        
        if node_features:
            x = torch.tensor(nodes[nodes.columns[:-1]].values, dtype=torch.float)
        else:
            x = torch.ones(nodes.shape[0], 1)
        if edge_features:
            edge_attr = torch.tensor(edges[edges.columns[:-1]].values, dtype=torch.float)
        if node_labels:
            y = torch.tensor(nodes[nodes.columns[-1]].values, dtype=torch.long)
        elif edge_labels:
            y = torch.tensor(edges[edges.columns[-1]].values, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # #look for nan values
        # print('Nan values in x:', torch.isnan(data.x).sum())
        # print('Nan values in edge_attr:', torch.isnan(data.edge_attr).sum())
        # print('Nan values in y:', torch.isnan(data.y).sum())
        # #inspect x nan values
        # print('Nan values in x:', data.x[torch.isnan(data.x)])


        # print('Data object', data)
        return data

    def get_data(self):
        return self.data

#trainset = AmlsimDataset('data/1bank/bank/trainset/nodes.csv', 'data/1bank/bank/trainset/edges.csv')
#traindata = trainset.get_data()
#print(traindata.num_features)
    

class AmlsimDatasetWithaccoount():
    def __init__(self, node_file:str, edge_file:str, node_features:bool=False, edge_features:bool=True, node_labels:bool=False, edge_labels:bool=False, seed:int=42):
        self.data = self.load_data(node_file, edge_file, node_features, edge_features, node_labels, edge_labels)
        
    def load_data(self, node_file, edge_file, node_features, edge_features, node_labels, edge_labels):
        nodes = pd.read_csv(node_file)
        edges = pd.read_csv(edge_file)
        edge_index = torch.tensor(edges[['src', 'dst']].values, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        
       #nodes = nodes.drop(columns=['bank', 'account'])

        nodes = nodes.fillna(0)
        
        if node_features:
            x = torch.tensor(nodes[nodes.columns[:-1]].values, dtype=torch.float)
        else:
            x = torch.ones(nodes.shape[0], 1)
        if edge_features:
            edge_attr = torch.tensor(edges[edges.columns[:-1]].values, dtype=torch.float)
        if node_labels:
            y = torch.tensor(nodes[nodes.columns[-1]].values, dtype=torch.long)
        elif edge_labels:
            y = torch.tensor(edges[edges.columns[-1]].values, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # #look for nan values
        # print('Nan values in x:', torch.isnan(data.x).sum())
        # print('Nan values in edge_attr:', torch.isnan(data.edge_attr).sum())
        # print('Nan values in y:', torch.isnan(data.y).sum())
        # #inspect x nan values
        # print('Nan values in x:', data.x[torch.isnan(data.x)])


        # print('Data object', data)
        return data

    def get_data(self):
        return self.data


def AMLSimDataset(tx_log_path:str, feature_engineering_method):
    # input:
    # tx_log_path:
    #   - path to the transaction log file
    # feature_engineering_method:
    #   - tx_log -> (x, y, edge_index) where x = node features, y = labels, edge_index = edge index
    #
    # output:
    # trainset:
    #   - (torch_geometric.data.Data): training dataset
    # valset:
    #   - (torch_geometric.data.Data): validation dataset
    # testset:
    #   - (torch_geometric.data.Data): test dataset
    
    # --- Loading data ---
    df = pd.read_csv(tx_log_path)
    n_steps = df['step'].max()

    # --- Temporal Train/Val/Test split ---
    set_size = n_steps // 3
    train_val_overlap = 0.2
    val_test_overlap = 0.2
    t0 = int(0)
    t1 = int(t0 + set_size + set_size*train_val_overlap*(2/3))
    t2 = int(t0 + set_size - set_size*train_val_overlap*(1/3))
    t3 = int(t0 + 2*set_size + set_size*val_test_overlap*(1/3))
    t4 = int(t0 + 2*set_size - set_size*val_test_overlap*(2/3))
    t5 = int(n_steps)
    train_interval = [t0, t1]
    val_interval = [t2, t3]
    test_interval = [t4, t5]
    df_train = df[(df['step'] >= train_interval[0]) & (df['step'] < train_interval[1])]
    df_val = df[(df['step'] >= val_interval[0]) & (df['step'] < val_interval[1])]
    df_test = df[(df['step'] >= test_interval[0]) & (df['step'] < test_interval[1])]
    
    # --- Feature Engineering ---
    x_train, edge_index_train, y_train = feature_engineering_method(df_train)
    x_val, edge_index_val, y_val= feature_engineering_method(df_val)
    x_test, edge_index_test, y_test = feature_engineering_method(df_test)
    
    # --- Normalization & Other Preprocessing ---
    # TODO (How to treat categorical variables?)
    
    # --- Convert to correct type ---
    trainset = Data(x=x_train, edge_index=edge_index_train, y=y_train)
    valset = Data(x=x_val, edge_index=edge_index_val, y=y_val)
    testset = Data(x=x_test, edge_index=edge_index_test, y=y_test)
    
    return trainset, valset, testset
