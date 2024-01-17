import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split
import os

class EllipticDataset():
    def __init__(self, data_folder, val_size=0.2, test_size=0.2, seed=42):
        # read in labels
        classes = pd.read_csv(data_folder + "/elliptic_txs_classes.csv")
        # read in edge pairs
        edges = pd.read_csv(data_folder + "/elliptic_txs_edgelist.csv")
        # read in features
        features = pd.read_csv(data_folder + "/elliptic_txs_features.csv", header=None)
        
        # remap class, licit: 0, illicit: 1, unknown: -1
        classes["class"] = classes["class"].map({"1": 1, "2": 0, "unknown": -1})

        # merge features and labels
        df = features.merge(classes, how="left", left_on=0, right_on="txId")
        df = df.sort_values(0).reset_index(drop=True)
        assert len(df) == len(classes)

        # drop unclassified and isolated nodes
        classified_nodes = set(classes[classes["class"] != -1]["txId"].values)
        assert len(classified_nodes) == 46564
        classified_edges = edges[(edges["txId1"].isin(classified_nodes)) & (edges["txId2"].isin(classified_nodes))].copy()
        non_isolated_nodes = set(classified_edges["txId1"].values).union(classified_edges["txId2"].values)
        classified_df = df[df[0].isin(non_isolated_nodes)].copy()
        
        # reindex nodes
        classified_df = classified_df.sort_values(1).reset_index(drop=True)
        old2new = {old:new for new, old in enumerate(classified_df[0].values)}
        classified_edges["txId1"] = classified_edges["txId1"].map(old2new)
        classified_edges["txId2"] = classified_edges["txId2"].map(old2new)
        classified_df[0] = classified_df[0].map(old2new)

        # edges
        edge_index = torch.tensor(classified_edges.values, dtype=torch.long)
        edge_index = edge_index.t().contiguous()

        # labels 
        labels = classified_df["class"].values
        labels = torch.tensor(labels, dtype=torch.float)

        # timestamps 
        timestamps = set(classified_df[1].values)

        # features
        features = torch.tensor(classified_df.drop([0, 1, "class", "txId"], axis=1).values, dtype=torch.float)

        # construct torch_geometric.data.Data
        self.data = Data(x=features, edge_index=edge_index, y=labels)
        
        # generate array of indices
        indices = np.arange(len(labels))

        # split indices into train, val, and test sets
        self.train_indices, test_indices, self.train_labels, test_labels = train_test_split(indices, labels, test_size=val_size+test_size, stratify=labels, random_state=42) 
        self.val_indices, self.test_indices, self.val_labels, self.test_labels = train_test_split(test_indices, test_labels, test_size=test_size/(val_size+test_size), stratify=test_labels, random_state=42)
    
    def get_data(self):
        return self.data, self.train_indices, self.val_indices, self.test_indices, self.train_labels, self.val_labels, self.test_labels


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