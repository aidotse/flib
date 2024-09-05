import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np
import os


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

#NEW
class AmlsimDataset1():
    def __init__(self, node_file:str, edge_file:str, node_features:bool=False, edge_features:bool=True, node_labels:bool=False, edge_labels:bool=False, seed:int=42):
        self.data = self.load_data(node_file, edge_file, node_features, edge_features, node_labels, edge_labels)
        self.seed = seed
        self.create_masks()

    def load_data(self, node_file, edge_file, node_features, edge_features, node_labels, edge_labels):
        nodes = pd.read_csv(node_file)
        edges = pd.read_csv(edge_file)
        edge_index = torch.tensor(edges[['src', 'dst']].values, dtype=torch.long).t().contiguous()
        
        if node_features:
            x = torch.tensor(nodes[nodes.columns[:-1]].values, dtype=torch.float)
        else:
            x = torch.ones(nodes.shape[0], 1)
        edge_attr = torch.tensor(edges[edges.columns[:-1]].values, dtype=torch.float) if edge_features else None
        y = torch.tensor(nodes[nodes.columns[-1]].values, dtype=torch.long) if node_labels else torch.tensor(edges[edges.columns[-1]].values, dtype=torch.long) if edge_labels else None
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def create_masks(self):
        # Assuming labels (`y`) are present and are for nodes
        num_nodes = self.data.num_nodes
        labels = self.data.y.numpy()  # Convert to numpy for easier handling

        # Generate indices for train/val/test
        idx = np.arange(num_nodes)
        idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=self.seed)
        idx_train, idx_val = train_test_split(idx_train, test_size=0.25, random_state=self.seed)  # 0.25 * 0.8 = 0.2

        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask

    def get_data(self):
        return self.data


class AmlsimDataset2():
    def __init__(self, node_file:str, edge_file:str, node_features:bool=False, edge_features:bool=True, node_labels:bool=False, edge_labels:bool=False, seed:int=42, train_test_split:float=0.8):
        self.seed = seed
        self.train_test_split = train_test_split
        self.data = self.load_data(node_file, edge_file, node_features, edge_features, node_labels, edge_labels)
        
    def load_data(self, node_file, edge_file, node_features, edge_features, node_labels, edge_labels):
        # Ensure reproducibility of the split
        torch.manual_seed(self.seed)
        
        nodes = pd.read_csv(node_file)
        edges = pd.read_csv(edge_file)
        
        edge_index = torch.tensor(edges[['src', 'dst']].values, dtype=torch.long).t().contiguous()
        
        if node_features:
            x = torch.tensor(nodes[nodes.columns[:-1]].values, dtype=torch.float)
        else:
            x = torch.ones((nodes.shape[0], 1), dtype=torch.float)
            
        edge_attr = torch.tensor(edges[edges.columns[:-1]].values, dtype=torch.float) if edge_features else None
        
        y = None
        if node_labels:
            y = torch.tensor(nodes[nodes.columns[-1]].values, dtype=torch.long)
        elif edge_labels:
            y = torch.tensor(edges[edges.columns[-1]].values, dtype=torch.long)
        
        # Creating masks
        num_nodes = nodes.shape[0]
        num_train = int(num_nodes * self.train_test_split)
        
        # Generate a permutation of indices
        perm = torch.randperm(num_nodes)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Use the first part for training
        train_mask[perm[:num_train]] = True
        # And the remainder for testing
        test_mask[perm[num_train:]] = True

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask, test_mask=test_mask)
        return data

    def get_data(self):
        return self.data
