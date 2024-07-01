from importlib import reload

import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split

import data
reload(data)

node_file = '/home/tomas/desktop/flib/thesis_XAML/data/100K_accts_MID5/bank/train/nodes.csv'

nodes = pd.read_csv(node_file)

num_samples = nodes[nodes['is_sar'] == 1].shape[0]
print("Number of samples with is_sar == 1:", num_samples)

print(nodes.head())