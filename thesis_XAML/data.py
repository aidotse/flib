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

        nodes = nodes.drop(columns=['bank']) #Needed for GNN training
        # #look for nan values in nodes
        # print('Nan values in nodes:', nodes.isnull().sum().sum())
        # #print sum of not-nan values in nodes
        # print('Not-nan values in nodes:', nodes.notnull().sum().sum())

        # #print examples rows with nan values
        # print('Examples of rows with nan values in nodes:')
        # print(nodes[nodes.isnull().any(axis=1)].head())
        # #print how many nan values there are in each column
        # print('Nan values in each column in nodes:')
        # print(nodes.isnull().sum())



        # print('Nodes type',type(nodes))
        # print('Nodes shape', nodes.shape)
        # print('values type',type(nodes[nodes.columns[:-1]].values))
        # print('values shape',nodes[nodes.columns[:-1]].values.shape)
        # print('chatgpt',nodes[nodes.columns[:-1]].dtypes)
        # print(nodes.head())

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