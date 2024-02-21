import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split
import os


# (Written by Edvin)
class AmlsimDataset():
    def __init__(self, node_file:str, edge_file:str, node_features:bool=False, edge_features:bool=True, node_labels:bool=False, edge_labels:bool=False, seed:int=42):
        self.data = self.load_data(node_file, edge_file, node_features, edge_features, node_labels, edge_labels)
        
    def load_data(self, node_file, edge_file, node_features, edge_features, node_labels, edge_labels):
        nodes = pd.read_csv(node_file)
        edges = pd.read_csv(edge_file)
        edge_index = torch.tensor(edges[['src', 'dst']].values, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        
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
        return data

    def get_data(self):
        return self.data

#trainset = AmlsimDataset('data/1bank/bank/trainset/nodes.csv', 'data/1bank/bank/trainset/edges.csv')
#traindata = trainset.get_data()
#print(traindata.num_features)