# General
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from importlib import reload
from types import SimpleNamespace
import shap
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.explain import Explainer, GNNExplainer
from sklearn.metrics import accuracy_score

# Custom
import utils
reload(utils)

import data
reload(data)

import train
reload(train)

import modules
reload(modules)
from modules import GCN, GCN_GNNExplainer, GCN_GraphSVX
from modules import GraphSAGE
from torch_geometric.data import DataLoader
import torch.optim as optim

import explanations
reload(explanations)
from explanations import LIME_explanation, SHAP_explanation

import GraphSVX_explainers
reload(GraphSVX_explainers)
from GraphSVX_explainers import GraphSVX, GraphLIME


# Choose model and dataset here
model_name = 'GAT'
dataset_name = '100K_accts_EASY25'

# Load model, data, and other variables
model_path = f'trained_models/{model_name}_{dataset_name}.pth'

checkpoint = torch.load(model_path)
model_state_dict = checkpoint['model_state_dict']
hyperparameters = checkpoint['hyperparameters']
traindata = checkpoint['traindata']
testdata = checkpoint['testdata']
#feature_names = checkpoint['feature_names']
feature_names = ['account', 'sum', 'mean', 'median', 'std', 'max', 'min', 'in_sum', 'out_sum', 'in_mean', 'out_mean', 'in_median', 'out_median', 'in_std', 'out_std', 'in_max', 'out_max', 'in_min', 'out_min', 'count_in', 'count_out', 'n_unique_in', 'n_unique_out', 'count_days_in_bank', 'count_phone_changes', 'sum_spending', 'mean_spending', 'median_spending', 'std_spending', 'max_spending', 'min_spending', 'count_spending']
target_names = checkpoint['target_names']
running_train_loss = checkpoint['running_train_loss']
running_test_loss = checkpoint['running_test_loss']
accuracy = checkpoint['accuracy']

# Initialize model
if model_name == 'GAT':
    in_channels = traindata.x.shape[1]
    hidden_channels = hyperparameters['hidden_channels']
    out_channels = 2
    num_heads = hyperparameters['num_heads']
    dropout = hyperparameters['dropout']
    model = modules.GAT_GraphSVX_foroptuna(in_channels = in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_heads=num_heads, dropout=dropout)

model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
traindata = traindata.to(device)
testdata = testdata.to(device)
model.eval()

# Set return type of model to log_probas for explainer compatibility
model.set_return_type('log_probas')


# Find the indices of the nodes that are predicted as SAR
model.eval()
with torch.no_grad():
    y_probas = model.forward(testdata.x, testdata.edge_index)
    y_probas = torch.exp(y_probas)
    y_pred = y_probas.cpu().numpy().argmax(axis=1)    
    y_true = testdata.y.cpu().numpy()

sar_indices = np.where(y_pred == 1)[0]

########################################
##### EXPLANATION PART STARTS HERE #####
########################################

# TODO:
# [x] Set number of samples according to number of nodes
# [x] Add placeholders for accuracy, n_sus_neighbours, threshold
# [x] Ändra i windows så att datorn inte går in i sleep, men att skärmen blir svart
# [x] Lägg till så att all information sparas i placeholders
# [x] Ta bort onödiga outputs
# [x] Lägg till output om progress
# [x] Gör loop över hela
# [x] Lägg till sparfunktion som sparar i slutet av varje 100e loop
# [x] Testa för 10 punkter
# [ ] Kör hela (Den stannade efter typ 3000 iterationer, tappade vpn-connection)


n_sar_indices = len(sar_indices)
print(f'Number of SAR nodes: {n_sar_indices}')

# Prepare placeholders for information to be computed
threshold = np.arange(0,1,0.01)
thresh_stop = np.zeros(n_sar_indices)
n_sus_neighbours = np.zeros((n_sar_indices,len(threshold)))
SVX_accuracy = np.zeros((n_sar_indices,len(threshold)))

print(threshold.shape)
print(n_sus_neighbours.shape)
print(SVX_accuracy.shape)
print(thresh_stop.shape)


#########################################
#### BÖRJA FOR LOOP HÄR #################
#########################################

time_start = time.time()


for i in range(n_sar_indices):
    os.system('cls' if os.name == 'nt' else 'clear')

    time_checkpoint = time.time()
    time_elapsed = time_checkpoint - time_start
    print(f'Progress: {i}/{n_sar_indices}.')
    print(f'Time elapsed: {time_elapsed/60:.2f} minutes.')

    # ------------ OBSERVE, HERE IS NODE TO EXPLAIN --------------------
    node_to_explain = sar_indices[i].item()
    # ---------------------------------------------------------

    subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(node_to_explain, 3, testdata.edge_index, relabel_nodes=False)
    org_to_new_mapping, new_to_org_mapping, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
    testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])

    n_nodes = subset_expl.shape[0]
    num_samples = 5*n_nodes

    # Running the explainer
    testdata_expl = testdata_expl.to('cpu')
    model = model.to('cpu')

    data = SimpleNamespace()
    data.x = testdata_expl.x
    data.edge_index  = testdata_expl.edge_index
    data.y = testdata_expl.y
    data.num_classes = 2
    data.num_features = 10
    data.num_nodes = testdata_expl.x.shape[0]
    data.name = 'test'

    explainer = GraphSVX(data, model, True)
    explanations = explainer.explain(node_indexes=[0], num_samples=num_samples, hops = 3, vizu = False) # NOTE: Here we use the number of nodes in the subgraph as the number of samples

    F = explainer.F
    SV = explanations[0]
    SV_nodes = SV[F:]

    import numpy as np

    for j, thresh in enumerate(threshold):
        suspicious_neighbours_orgidx = []
        suspicious_neighbours_sv = []
        suspicious_neighbours_ypred = []
        suspicious_neighbours_y = []
        
        for node_idx, sv in enumerate(SV_nodes):
            
            node_idx = node_idx + 1 #sv at position 0 corresponds to node 1 (since node 0 is not assigned a Shapley value)
            if sv > thresh:
                #suspicious_neighbours_orgidx.append(new_to_org_mapping[node_idx].item())
                suspicious_neighbours_orgidx.append(node_idx)
                suspicious_neighbours_sv.append(sv)
                suspicious_neighbours_ypred.append(1)
                suspicious_neighbours_y.append(testdata_expl.y[node_idx].item())

        if len(suspicious_neighbours_orgidx) == 0:
            thresh_stop[i] = thresh
            break
        
        SVX_accuracy[i,j] = accuracy_score(suspicious_neighbours_y, suspicious_neighbours_ypred)
        n_sus_neighbours[i,j] = len(suspicious_neighbours_ypred)

        if i % 100 == 0:
            np.save('/home/tomas/desktop/flib/thesis_XAML/SVX_accuracy_save/threshold.npy', threshold)
            np.save('/home/tomas/desktop/flib/thesis_XAML/SVX_accuracy_save/n_sus_neighbours.npy', n_sus_neighbours)
            np.save('/home/tomas/desktop/flib/thesis_XAML/SVX_accuracy_save/SVX_accuracy.npy', SVX_accuracy)
            np.save('/home/tomas/desktop/flib/thesis_XAML/SVX_accuracy_save/thresh_stop.npy', thresh_stop)

time_stop = time.time()
time_elapsed = time_stop - time_start
os.system('cls' if os.name == 'nt' else 'clear')
print('Done. Time elapsed: {:.2f} minutes.'.format(time_elapsed/60))

