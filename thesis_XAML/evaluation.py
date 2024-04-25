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

def get_all_SARS_feature_importance():
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
    feature_names = ['sum', 'mean', 'median', 'std', 'max', 'min', 'in_sum', 'out_sum', 'in_mean', 'out_mean', 'in_median', 'out_median', 'in_std', 'out_std', 'in_max', 'out_max', 'in_min', 'out_min', 'count_in', 'count_out', 'n_unique_in', 'n_unique_out', 'count_days_in_bank', 'count_phone_changes', 'sum_spending', 'mean_spending', 'median_spending', 'std_spending', 'max_spending', 'min_spending', 'count_spending']
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

    n_sar_indices = len(sar_indices)
    print(f'Number of SAR nodes: {n_sar_indices}')
    
    
    feature_importance_matrix_GraphSVX = np.zeros((n_sar_indices, len(feature_names)))
    feature_importance_matrix_LIME = np.zeros((n_sar_indices, len(feature_names)))
    feature_importance_matrix_SHAP = np.zeros((n_sar_indices, len(feature_names)))
    feature_name_to_index_dict = {feature_name : i for i, feature_name in enumerate(feature_names)}
    

    time_start = time.time()

    for i in range(10):#range(n_sar_indices):
        os.system('cls' if os.name == 'nt' else 'clear')

        time_checkpoint = time.time()
        time_elapsed = time_checkpoint - time_start
        print(f'Progress: {i}/{n_sar_indices}.')
        print(f'Time elapsed: {time_elapsed/60:.2f} minutes.')
        if i > 0:
            print(f'Last iteration: Time spent in SVX = {time_spent_SVX:.2f} seconds | Time spent in LIME = {time_spent_LIME:.2f} seconds | Time spent in SHAP = {time_spent_SHAP:.2f} seconds.')

        # ------------ OBSERVE, HERE IS NODE TO EXPLAIN --------------------
        node_to_explain = sar_indices[i].item()
        # ------------------------------------------------------------------

        print('Running GraphSVX')
        model.set_return_type('log_probas')
        subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(node_to_explain, 3, testdata.edge_index, relabel_nodes=False)
        org_to_new_mapping, new_to_org_mapping, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
        testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])

        n_nodes = subset_expl.shape[0]
        num_samples = 4*n_nodes

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

        time_start_SVX = time.time()
        explainer = GraphSVX(data, model, True)
        explanations = explainer.explain(node_indexes=[0], num_samples=num_samples, hops = 3, vizu = False) # NOTE: Here we use the number of nodes in the subgraph as the number of samples

        F = explainer.F
        D = explanations[0].shape[0] - F
        SV = explanations[0]
        SV_features = SV[:F]
        SV_nodes = SV[F:]
        time_stop_SVX = time.time()
        time_spent_SVX = time_stop_SVX - time_start_SVX
        
        model = model.to(device)
        testdata_expl = testdata_expl.to(device)

        # --- LIME ---
        print('Running LIME')
        num_features = 10
        class_prob_fn = model.forward_NFVinput

        # Prepare the model for handling the LIME explainer
        model.set_test_data(testdata_expl)
        model.set_node_to_explain(0)

        time_start_LIME = time.time()
        exp_LIME = LIME_explanation(node_to_explain = node_to_explain,
                                            num_features = num_features,
                                            class_prob_fn = class_prob_fn,
                                            testdata = testdata,
                                            feature_names = feature_names,
                                            target_names = target_names)

        exp_LIME.show_in_notebook(show_table=True, show_all=False)
        time_stop_LIME = time.time()
        time_spent_LIME = time_stop_LIME - time_start_LIME
        
        # --- SHAP ---
        print('Running SHAP')
        K = 20
        class_prob_fn = model.forward_NFVinput

        time_start_SHAP = time.time()
        exp_SHAP = SHAP_explanation(node_to_explain = node_to_explain,
                                                class_prob_fn = class_prob_fn,
                                                backgrounddata = traindata,
                                                explaindata = testdata,
                                                feature_names = feature_names,
                                                K = K)

        shap.plots.waterfall(exp_SHAP[0])
        time_stop_SHAP = time.time()
        time_spent_SHAP = time_stop_SHAP - time_start_SHAP
        
        # Create dictionary with feature importance for GraphSVX
        feat_idx, discarded_feat_idx = explainer.feature_selection(0, "Expectation") # Using 0 here since 0 is node_to_explain in the subgraph
        print(len(feat_idx))
        feature_names_in_explanation = [feature_names[i] for i in feat_idx]
        
        feature_importance_GraphSVX = {feat_name : feat_val for feat_name, feat_val in zip(feature_names_in_explanation, SV_features)}
        for feat_name in feature_importance_GraphSVX.keys():
            print(f'{feat_name}: {feature_importance_GraphSVX[feat_name]:.4f}')
            feature_importance_matrix_GraphSVX[i, feature_name_to_index_dict[feat_name]] = feature_importance_GraphSVX[feat_name]
        print(f'feature_importance_matrix_GraphSVX: {feature_importance_matrix_GraphSVX[i, :]}')
        
        print()
        
        # Create dictionary with feature importance for LIME:
        feature_importance_LIME = {feature_names[feat_index] : feat_val for (feat_index, feat_val) in exp_LIME.local_exp[1]}
        for feat_name in feature_importance_LIME.keys():
            print(f'{feat_name}: {feature_importance_LIME[feat_name]:.4f}')
            feature_importance_matrix_LIME[i, feature_name_to_index_dict[feat_name]] = feature_importance_LIME[feat_name]
        print(f'feature_importance_matrix_LIME: {feature_importance_matrix_LIME[i, :]}')

        print()

        # Create dictionary with feature importance for SHAP:
        feature_importance_SHAP = {feat_name : feat_val for feat_name, feat_val in zip(feature_names, exp_SHAP[0].values)}
        for feat_name in feature_importance_SHAP.keys():
            print(f'{feat_name}: {feature_importance_SHAP[feat_name]:.4f}')
            feature_importance_matrix_SHAP[i, feature_name_to_index_dict[feat_name]] = feature_importance_SHAP[feat_name]
        print(f'feature_importance_matrix_SHAP: {feature_importance_matrix_SHAP[i, :]}')
    
        # Save feature importance matrices after every 100th iteration
        if i % 100 == 0 or i == 10:
            np.save('feature_importance_GraphSVX.npy', feature_importance_matrix_GraphSVX)
            np.save('feature_importance_LIME.npy', feature_importance_matrix_LIME)
            np.save('feature_importance_SHAP.npy', feature_importance_matrix_SHAP)
    
    return feature_importance_matrix_GraphSVX, feature_importance_matrix_LIME, feature_importance_matrix_SHAP

def load_all_SARS_feature_importances():
    feature_importance_matrix_GraphSVX = np.load('feature_importance_GraphSVX.npy')
    feature_importance_matrix_LIME = np.load('feature_importance_LIME.npy')
    feature_importance_matrix_SHAP = np.load('feature_importance_SHAP.npy')
    return feature_importance_matrix_GraphSVX, feature_importance_matrix_LIME, feature_importance_matrix_SHAP

def correctness():
    print('Not implemented yet.')

def completeness():
    print('Not implemented yet.')

def compactness():
    print('Not implemented yet.')

def confidence():
    print('Not implemented yet.')

def coherence():
    print('Not implemented yet.')
    
    feature_importance_matrix_GraphSVX, feature_importance_matrix_LIME, feature_importance_matrix_SHAP = load_all_SARS_feature_importances()
    
    k = 5
    ind_topk_SVX = np.argpartition(abs(feature_importance_matrix_GraphSVX), range(-k, 0), axis=1)[:, :-(k+1):-1]
    ind_topk_LIME = np.argpartition(abs(feature_importance_matrix_LIME), range(-k, 0), axis=1)[:, :-(k+1):-1]
    ind_topk_SHAP = np.argpartition(abs(feature_importance_matrix_SHAP), range(-k, 0), axis=1)[:, :-(k+1):-1]
    print(f'ind_topk_SVX: {ind_topk_SVX}')
    print(f'ind_topk_LIME: {ind_topk_LIME}')
    print(f'ind_topk_SHAP: {ind_topk_SHAP}')
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot for ind_topk_SVX
    axs[0].bar(range(k), ind_topk_SVX[0])
    axs[0].set_title('ind_topk_SVX')
    axs[0].set_xlabel('Feature Index')
    axs[0].set_ylabel('Rank')

    # Plot for ind_topk_LIME
    axs[1].bar(range(k), ind_topk_LIME[0])
    axs[1].set_title('ind_topk_LIME')
    axs[1].set_xlabel('Feature Index')
    axs[1].set_ylabel('Rank')

    # Plot for ind_topk_SHAP
    axs[2].bar(range(k), ind_topk_SHAP[0])
    axs[2].set_title('ind_topk_SHAP')
    axs[2].set_xlabel('Feature Index')
    axs[2].set_ylabel('Rank')
    
    plt.savefig('/home/tomas/desktop/flib/thesis_XAML/coherence_plot.png')

    
    # Feature agreement
    # Rank agreement
    # Sign agreement
    # Signed rank agreement

def main():
    print('Running experiments for evaluation of SVX.')
    #get_all_SARS_feature_importance()
    coherence()
    # Plotting

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()