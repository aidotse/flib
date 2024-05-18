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
import pickle
import multiprocessing

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


WD_PATH = os.path.dirname(os.path.abspath(__file__))


def get_sar_indices(model, testdata):
    # Find the indices of the nodes that are predicted as SAR
    model.set_return_type('log_probas')
    model.eval()
    with torch.no_grad():
        y_probas = model.forward(testdata.x, testdata.edge_index)
        y_probas = torch.exp(y_probas)
        y_pred = y_probas.cpu().numpy().argmax(axis=1)    

    sar_indices = np.where(y_pred == 1)[0]
    
    return sar_indices


def load_model(model_name, dataset_name):
    
    # Load model, data, and other variables
    model_path = f'/home/tomas/desktop/flib/thesis_XAML/trained_models/{model_name}_{dataset_name}.pth'
    
    checkpoint = torch.load(model_path)
    model_state_dict = checkpoint['model_state_dict']
    hyperparameters = checkpoint['hyperparameters']
    traindata = checkpoint['traindata']
    testdata = checkpoint['testdata']
    feature_names = ['sum', 'mean', 'median', 'std', 'max', 'min', 'in_sum', 'out_sum', 'in_mean', 'out_mean', 'in_median', 'out_median', 'in_std', 'out_std', 'in_max', 'out_max', 'in_min', 'out_min', 'count_in', 'count_out', 'n_unique_in', 'n_unique_out', 'count_days_in_bank', 'count_phone_changes', 'sum_spending', 'mean_spending', 'median_spending', 'std_spending', 'max_spending', 'min_spending', 'count_spending']
    target_names = checkpoint['target_names']

    # Initialize model
    if model_name == 'GAT':
        in_channels = traindata.x.shape[1]
        hidden_channels = hyperparameters['hidden_channels']
        out_channels = 2
        num_heads = hyperparameters['num_heads']
        dropout = hyperparameters['dropout']
        model = modules.GAT_GraphSVX_foroptuna(in_channels = in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_heads=num_heads, dropout=dropout)
    elif model_name == 'GraphSAGE':
        in_channels = traindata.x.shape[1]
        hidden_channels = hyperparameters['hidden_channels']
        out_channels = 2
        dropout = hyperparameters['dropout']
        model = modules.GraphSAGE_GraphSVX_foroptuna(in_channels = in_channels, hidden_channels=hidden_channels, out_channels=out_channels, dropout=dropout)
    else:
        raise ValueError("model_name must be either 'GraphSAGE' or 'GAT'")
    
    # Set model parameters
    model.load_state_dict(model_state_dict)
    
    # Move to device and set model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    traindata = traindata.to(device)
    testdata = testdata.to(device)
    model.eval()

    # Set return type of model to log_probas for explainer compatibility
    model.set_return_type('log_probas')

    return model, traindata, testdata, feature_names, target_names


def calculate_feature_importance_SVX(model_name, dataset_name):
    # Set saving directory
    data_save_dir = f'{WD_PATH}/evaluation_data/feature_importance'
    if not os.path.isdir(data_save_dir):
        os.makedirs(data_save_dir)
    
    # Load model
    model, _, testdata, feature_names, _ = load_model(model_name, dataset_name)
    
    # Calculate SAR indices
    sar_indices = get_sar_indices(model, testdata)
    n_sar_indices = len(sar_indices)
    
    # Run explainer over all SAR indices
    model.eval()
    model.set_return_type('log_probas')    
    
    # Placeholders for feature importance and r2 matrices
    feature_importance_GraphSVX = np.zeros((n_sar_indices, len(feature_names)))
    node_importance_GraphSVX = list(range(n_sar_indices))
    r2_GraphSVX = np.zeros((n_sar_indices,1))
    feature_name_to_index_dict = {feature_name : i for i, feature_name in enumerate(feature_names)}
    
    time_start = time.time()
    for i in range(n_sar_indices):
        os.system('cls' if os.name == 'nt' else 'clear')

        time_checkpoint = time.time()
        time_elapsed = time_checkpoint - time_start
        print('Running GraphSVX')
        print(f'Progress: {i}/{n_sar_indices}.')
        print(f'Time elapsed: {time_elapsed/60:.2f} minutes.')
        if i > 0:
            print(f'Last iteration: Time spent in SVX = {time_spent_SVX:.2f} seconds')

        # ------------ OBSERVE, HERE IS NODE TO EXPLAIN --------------------
        node_to_explain = sar_indices[i].item()
        # ------------------------------------------------------------------

        time_start_SVX = time.time()
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
        data.num_nodes = testdata_expl.x.shape[0]
        data.name = 'test'

        explainer = GraphSVX(data, model, True)
        explanations, r2_SVX = explainer.explain(node_indexes=[0], num_samples=num_samples, hops = 3, vizu = False, return_r2 = True)

        F = explainer.F
        SV = explanations[0]
        SV_features = SV[:F]
        SV_nodes = SV[F:]
        time_stop_SVX = time.time()
        time_spent_SVX = time_stop_SVX - time_start_SVX
        
        # Save feature importance every 100th iteration
        feat_idx, _ = explainer.feature_selection(0, "Expectation") # Using 0 here since 0 is node_to_explain in the subgraph

        feature_names_in_explanation = [feature_names[i] for i in feat_idx]
        
        feature_importance_GraphSVX_dict = {feat_name : feat_val for feat_name, feat_val in zip(feature_names_in_explanation, SV_features)}
        for feat_name in feature_importance_GraphSVX_dict.keys():
            feature_importance_GraphSVX[i, feature_name_to_index_dict[feat_name]] = feature_importance_GraphSVX_dict[feat_name]
        
        node_importance_GraphSVX[i] = (SV_nodes, org_to_new_mapping, new_to_org_mapping)
        
        r2_GraphSVX[i] = r2_SVX
        
        if i % 100 == 0 or i == 10 or i == n_sar_indices-1:
            with open(f'{data_save_dir}/{model_name}_{dataset_name}_feature_importance_GraphSVX.pkl', 'wb') as f:
                pickle.dump(feature_importance_GraphSVX, f)
            with open(f'{data_save_dir}/{model_name}_{dataset_name}_node_importance_GraphSVX.pkl', 'wb') as f:
                pickle.dump(node_importance_GraphSVX, f)
            with open(f'{data_save_dir}/{model_name}_{dataset_name}_r2_GraphSVX.pkl', 'wb') as f:
                pickle.dump(r2_GraphSVX, f)
            # np.save(f'evaluation_data/{model_name}_{dataset_name}_feature_importance_GraphSVX.npy', feature_importance_GraphSVX)
            # np.save(f'evaluation_data/{model_name}_{dataset_name}_node_importance_GraphSVX.npy', node_importance_GraphSVX)
            # np.save(f'evaluation_data/{model_name}_{dataset_name}_r2_GraphSVX.npy', r2_GraphSVX)
            
    print('Finished.')


def calculate_feature_importance_LIME(model_name, dataset_name):
    # Set saving directory
    data_save_dir = f'{WD_PATH}/evaluation_data/feature_importance'
    if not os.path.isdir(data_save_dir):
        os.makedirs(data_save_dir)
    
    # Load model
    model, _, testdata, feature_names, target_names = load_model(model_name, dataset_name)
    
    # Calculate SAR indices
    sar_indices = get_sar_indices(model, testdata)
    n_sar_indices = len(sar_indices)
    
    # Run explainer over all SAR indices
    model.eval()
    model.set_return_type('logits')
    
    # Placeholders for feature importance
    feature_importance_LIME = np.zeros((n_sar_indices, len(feature_names)))
    r2_LIME = np.zeros((n_sar_indices,1))
    feature_name_to_index_dict = {feature_name : i for i, feature_name in enumerate(feature_names)}
    
    time_start = time.time()
    for i in range(n_sar_indices):
        os.system('cls' if os.name == 'nt' else 'clear')
        
        time_checkpoint = time.time()
        time_elapsed = time_checkpoint - time_start
        print('Running LIME')
        print(f'Progress: {i}/{n_sar_indices}.')
        print(f'Time elapsed: {time_elapsed/60:.2f} minutes.')
        if i > 0:
            print(f'Last iteration: Time spent in LIME = {time_spent_LIME:.2f} seconds')

        # ------------ OBSERVE, HERE IS NODE TO EXPLAIN --------------------
        node_to_explain = sar_indices[i].item()
        # ------------------------------------------------------------------

        time_start_LIME = time.time()
        subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(node_to_explain, 3, testdata.edge_index, relabel_nodes=False)
        _, _, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
        testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        testdata_expl = testdata_expl.to(device)

        # --- LIME ---
        print('Running LIME')
        num_features = 31
        class_prob_fn = model.forward_NFVinput

        # Prepare the model for handling the LIME explainer
        model.set_test_data(testdata_expl)
        model.set_node_to_explain(0)

        exp_LIME = LIME_explanation(node_to_explain = node_to_explain,
                                            num_features = num_features,
                                            class_prob_fn = class_prob_fn,
                                            testdata = testdata,
                                            feature_names = feature_names,
                                            target_names = target_names)

        #exp_LIME.show_in_notebook(show_table=True, show_all=False)
        time_stop_LIME = time.time()
        time_spent_LIME = time_stop_LIME - time_start_LIME
        
        feature_importance_LIME_dict = {feature_names[feat_index] : feat_val for (feat_index, feat_val) in exp_LIME.local_exp[1]}
        for feat_name in feature_importance_LIME_dict.keys():
            feature_importance_LIME[i, feature_name_to_index_dict[feat_name]] = feature_importance_LIME_dict[feat_name]
        
        r2_LIME[i] = exp_LIME.score
        
        if i % 100 == 0 or i == 10 or i == n_sar_indices-1:
            with open(f'{data_save_dir}/{model_name}_{dataset_name}_feature_importance_LIME.pkl', 'wb') as f:
                pickle.dump(feature_importance_LIME, f)
            with open(f'{data_save_dir}/{model_name}_{dataset_name}_r2_LIME.pkl', 'wb') as f:
                pickle.dump(r2_LIME, f)
    print('Finished.')


def calculate_feature_importance_SHAP(model_name, dataset_name):
    # Set saving directory
    data_save_dir = f'{WD_PATH}/evaluation_data/feature_importance'
    if not os.path.isdir(data_save_dir):
        os.makedirs(data_save_dir)
    
    # Load model
    model, traindata, testdata, feature_names, target_names = load_model(model_name, dataset_name)
    
    # Calculate SAR indices
    sar_indices = get_sar_indices(model, testdata)
    n_sar_indices = len(sar_indices)
    
    # Run explainer over all SAR indices
    model.eval()
    model.set_return_type('logits')
    
    # Placeholders for feature importance
    feature_importance_SHAP = np.zeros((n_sar_indices, len(feature_names)))
    feature_name_to_index_dict = {feature_name : i for i, feature_name in enumerate(feature_names)}
    
    time_start = time.time()
    for i in range(n_sar_indices):
        os.system('cls' if os.name == 'nt' else 'clear')

        time_checkpoint = time.time()
        time_elapsed = time_checkpoint - time_start
        print('Running SHAP')
        print(f'Progress: {i}/{n_sar_indices}.')
        print(f'Time elapsed: {time_elapsed/60:.2f} minutes.')
        if i > 0:
            print(f'Last iteration: Time spent in LIME = {time_spent_SHAP:.2f} seconds')

        # ------------ OBSERVE, HERE IS NODE TO EXPLAIN --------------------
        node_to_explain = sar_indices[i].item()
        # ------------------------------------------------------------------

        time_start_SHAP = time.time()
        subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(node_to_explain, 3, testdata.edge_index, relabel_nodes=False)
        _, _, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
        testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        testdata_expl = testdata_expl.to(device)

        # --- SHAP ---
        K = 20
        class_prob_fn = model.forward_NFVinput

        # Prepare the model for handling the SHAP explainer
        model.set_test_data(testdata_expl)
        model.set_node_to_explain(0)

        time_start_SHAP = time.time()
        exp_SHAP = SHAP_explanation(node_to_explain = node_to_explain,
                                                class_prob_fn = class_prob_fn,
                                                backgrounddata = traindata,
                                                explaindata = testdata,
                                                feature_names = feature_names,
                                                K = K)

        #shap.plots.waterfall(exp_SHAP[0])
        
        time_stop_SHAP = time.time()
        time_spent_SHAP = time_stop_SHAP - time_start_SHAP

        feature_importance_SHAP_dict = {feat_name : feat_val for feat_name, feat_val in zip(feature_names, exp_SHAP[0].values)}
        for feat_name in feature_importance_SHAP_dict.keys():
            feature_importance_SHAP[i, feature_name_to_index_dict[feat_name]] = feature_importance_SHAP_dict[feat_name]
        
        if i % 100 == 0 or i == 10 or i == n_sar_indices-1:
            with open(f'{data_save_dir}/{model_name}_{dataset_name}_feature_importance_SHAP.pkl', 'wb') as f:
                pickle.dump(feature_importance_SHAP, f)
    print('Finished.')


def load_evaluation_data_SVX(model_name, dataset_name):
    with open(f'{WD_PATH}/evaluation_data/feature_importance/{model_name}_{dataset_name}_feature_importance_GraphSVX.pkl', 'rb') as f:
        feature_importance_SVX = pickle.load(f)
    with open(f'{WD_PATH}/evaluation_data/feature_importance/{model_name}_{dataset_name}_node_importance_GraphSVX.pkl', 'rb') as f:
        node_importance_SVX = pickle.load(f)
    with open(f'{WD_PATH}/evaluation_data/feature_importance/{model_name}_{dataset_name}_r2_GraphSVX.pkl', 'rb') as f:
        r2_SVX = pickle.load(f)

    return feature_importance_SVX, node_importance_SVX, r2_SVX


def load_evaluation_data_LIME(model_name, dataset_name):
    with open(f'{WD_PATH}/evaluation_data/feature_importance/{model_name}_{dataset_name}_feature_importance_LIME.pkl', 'rb') as f:
        feature_importance_LIME = pickle.load(f)
    with open(f'{WD_PATH}/evaluation_data/feature_importance/{model_name}_{dataset_name}_r2_LIME.pkl', 'rb') as f:
        r2_LIME = pickle.load(f)
        
    return feature_importance_LIME, r2_LIME


def load_evaluation_data_SHAP(model_name, dataset_name):
    with open(f'{WD_PATH}/evaluation_data/feature_importance/{model_name}_{dataset_name}_feature_importance_SHAP.pkl', 'rb') as f:
        feature_importance_SHAP = pickle.load(f)
    return feature_importance_SHAP


# --- Functions for calculating coherence metrics ---
def TEST_ind_top(feature_importance, ind_top, n_instances):
    # Checking that the indices in ind_top are ordered correctly
    test_passed = True
    for i, instance in enumerate(feature_importance[:n_instances]):
        tmp_instance = np.copy(instance)
        tmp_instance = -np.sort(-np.abs(tmp_instance))
        for j,ind in enumerate(ind_top[i]):
            for k in range(j+1, len(tmp_instance)):
                if abs(instance[ind]) < tmp_instance[k]:
                    print('instance',abs(instance[ind]))
                    print('tmp_instance',tmp_instance[k])
                    print(f'Error in instance {i}, feature {ind}.')
                    test_passed = False
            if test_passed == False:
                break
        if test_passed == False:
            break
    if test_passed == True:
        print('TEST PASSED')
    else:
        print('TEST FAILED')


def TEST_get_topk_pairwise_feature_agreement():
    print('todo')


def TEST_get_topk_pairwise_rank_agreement():
    print('todo')


def TEST_get_topk_pairwise_sign_agreement():
    print('todo')


def get_ind_topk(feature_importance, k, importance_order = 'abs'):
    if importance_order == 'abs':
        ind_topk = np.argpartition(abs(feature_importance), range(-k, 0), axis=1)[:, :-(k+1):-1]
    elif importance_order == 'noabs':
        ind_topk = np.argpartition(feature_importance, range(-k, 0), axis=1)[:, :-(k+1):-1]
    else:
        raise ValueError('importance_order must be either "abs" or "noabs"')
    return ind_topk


def get_topk_pairwise_agreement(feature_importance_exp1, feature_importance_exp2, k, agreement_type, importance_order = 'abs'):
    n_samples = feature_importance_exp1.shape[0]    
    n_features = feature_importance_exp1.shape[1]
    
    # Get indices for the largest (absolute value) feature importances
    ind_topall_exp1 = get_ind_topk(feature_importance_exp1, n_features, importance_order)
    ind_topall_exp2 = get_ind_topk(feature_importance_exp2, n_features, importance_order)
    
    # Extract top-k indices from both explanations
    ind_topk_exp1 = ind_topall_exp1[:, :k].reshape((-1,k))
    ind_topk_exp2 = ind_topall_exp2[:, :k].reshape((-1,k))
    
    # Create placeholder for 'agreement score' (for example (feature agreement): percentage of overlap between the topk sets)
    agreement_score = np.zeros((n_samples,1))
    
    # Calculate agreement score
    if agreement_type == 'feature':
        
        for i in range(n_samples):
            ind_topk_common = np.intersect1d(ind_topk_exp1[i], ind_topk_exp2[i])
            agreement_score[i] = len(ind_topk_common)/k
            
    elif agreement_type == 'rank':
        
        for i in range(n_samples):
            ind_topk_common = np.intersect1d(ind_topk_exp1[i], ind_topk_exp2[i])
            count = 0
            for ind in ind_topk_common:
                if np.where(ind_topall_exp1[i] == ind)[0] == np.where(ind_topall_exp2[i] == ind)[0]:
                    count = count + 1
            agreement_score[i] = count/k
        
    elif agreement_type == 'sign':

        for i in range(n_samples):
            ind_topk_common = np.intersect1d(ind_topk_exp1[i], ind_topk_exp2[i])
            count = 0
            for ind in ind_topk_common:
                if np.sign(feature_importance_exp1[i, ind]) == np.sign(feature_importance_exp2[i, ind]):
                    count = count + 1
            agreement_score[i] = count/k
        
    elif agreement_type == 'signed_rank':
        
        for i in range(n_samples):
            ind_topk_common = np.intersect1d(ind_topk_exp1[i], ind_topk_exp2[i])
            count = 0
            for ind in ind_topk_common:
                is_sign_agreement = np.sign(feature_importance_exp1[i, ind]) == np.sign(feature_importance_exp2[i, ind])
                is_rank_agreement = np.where(ind_topall_exp1[i] == ind)[0] == np.where(ind_topall_exp2[i] == ind)[0]
                if is_sign_agreement and is_rank_agreement:
                    count = count + 1
            agreement_score[i] = count/k
        
    else:
        raise ValueError("agreement_type must be either 'feature', 'rank', 'sign', or 'signed_rank'")
    
    avg_agreement = np.mean(agreement_score)
    std_agreement = np.std(agreement_score)
    
    return [avg_agreement, std_agreement]


def get_all_topk_pairwise_agreement(feature_importance_exp1, feature_importance_exp2, agreement_type, importance_order = 'abs'):
    # Returns a list with elements of the form [avg, std] for feature agreement for all k's
    n_features = feature_importance_exp1.shape[1]
    
    # One data point for each k
    avg_std_agreement = np.zeros((n_features+1,2))
    avg_std_agreement[0] = [1,0]
    
    for k in range(1,n_features+1):
        avg_std_agreement[k] = get_topk_pairwise_agreement(feature_importance_exp1, feature_importance_exp2, k, agreement_type, importance_order)
    
    return avg_std_agreement


def get_agreement(feature_importance_list: list, agreement_type: str, importance_order = 'abs'):
    # Calculates all topk pairwise feature agreement between all explainer pairs (in the upper right triangle of the (n_explainers x n_explainers) table)
    n_explainers = len(feature_importance_list)
    n_comparisons = n_explainers*(n_explainers+1)//2

    agreement_list = list(range(n_comparisons))
    
    index = 0
    for i in range(n_explainers):
        for j in range(i, n_explainers):
            feature_importance_exp1 = feature_importance_list[i]
            feature_importance_exp2 = feature_importance_list[j]
            agreement_list[index] = get_all_topk_pairwise_agreement(feature_importance_exp1, feature_importance_exp2, agreement_type, importance_order)
            index = index + 1
            print(f'{agreement_type} agreement between explainer {i} and explainer {j} done.')
    
    return agreement_list


def create_and_save_agreement_plots(agreement_list: list, explainer_names: list, model_name: str, dataset_name: str, agreement_type: str, evaluation_type: str):
    if agreement_type != 'feature' and agreement_type != 'rank' and agreement_type != 'sign' and agreement_type != 'signed_rank':
        raise ValueError("agreement_type must be either 'feature', 'rank', 'sign', or 'signed_rank'")
    n_explainers = len(explainer_names)
    index = 0
    for i in range(n_explainers):
        for j in range(i,n_explainers):
            agreement_measure = agreement_list[index]
            avg = agreement_measure[:,0]
            std = agreement_measure[:,1]
            plt.figure()
            plt.plot(range(len(avg)), avg)
            plt.fill_between(range(len(avg)), avg-std, avg+std, alpha=0.3)
            plt.xlabel('k')
            plt.ylabel('Agreement (%)')
            plt.title(f'{agreement_type} Agreement between {explainer_names[i]} and {explainer_names[j]}')
            plt.show()
            
            save_dir = f'{WD_PATH}/evaluation_figures/{model_name}/{dataset_name}/{evaluation_type}'
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            
            plt.savefig(f'{save_dir}/{agreement_type}_agreement_{explainer_names[i]}_{explainer_names[j]}.png')
            plt.close()
            
            index = index + 1


def TEST_coherence():
    print('--- First test ---')
    # feature agreement = [0, 1]
    # rank agreement = [0, 0]
    # sign agreement = [0,0]
    # signed rank agreement = [0, 0]
    feature_importance_exp1 = np.array([0.2, 0.1]).reshape((1,2))
    feature_importance_exp2 = np.array([-0.1, -0.2]).reshape((1,2))
    
    feature_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'feature')
    feature_agreement_exp1_exp2 = feature_agreement_list[1]
    
    rank_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'rank')
    rank_agreement_exp1_exp2 = rank_agreement_list[1]
    
    sign_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'sign')
    sign_agreement_exp1_exp2 = sign_agreement_list[1]
    
    signed_rank_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'signed_rank')
    signed_rank_agreement_exp1_exp2 = signed_rank_agreement_list[1]

    #print(feature_agreement_list)
    
    if feature_agreement_exp1_exp2[1,0] == 0 and feature_agreement_exp1_exp2[2,0] == 1:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if rank_agreement_exp1_exp2[1,0] == 0 and rank_agreement_exp1_exp2[2,0] == 0:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if sign_agreement_exp1_exp2[1,0] == 0 and sign_agreement_exp1_exp2[2,0] == 0:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if signed_rank_agreement_exp1_exp2[1,0] == 0 and signed_rank_agreement_exp1_exp2[2,0] == 0:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    
    print('--- Second test ---')
    # feature agreement     = [0, 1]
    # rank agreement        = [0, 0]
    # sign agreement        = [0, 1]
    # signed rank agreement = [0, 0]
    feature_importance_exp1 = np.array([0.5, -0.1]).reshape((1,2))
    feature_importance_exp2 = np.array([0.2, -0.3]).reshape((1,2))
    
    feature_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'feature')
    feature_agreement_exp1_exp2 = feature_agreement_list[1]
    
    rank_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'rank')
    rank_agreement_exp1_exp2 = rank_agreement_list[1]
    
    sign_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'sign')
    sign_agreement_exp1_exp2 = sign_agreement_list[1]
    
    signed_rank_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'signed_rank')
    signed_rank_agreement_exp1_exp2 = signed_rank_agreement_list[1]

    #print(feature_agreement_list)
    
    if feature_agreement_exp1_exp2[1,0] == 0 and feature_agreement_exp1_exp2[2,0] == 1:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if rank_agreement_exp1_exp2[1,0] == 0 and rank_agreement_exp1_exp2[2,0] == 0:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if sign_agreement_exp1_exp2[1,0] == 0 and sign_agreement_exp1_exp2[2,0] == 1:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if signed_rank_agreement_exp1_exp2[1,0] == 0 and signed_rank_agreement_exp1_exp2[2,0] == 0:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    
    print('--- Third test ---')
    # feature agreement     = [1, 1]
    # rank agreement        = [1, 1]
    # sign agreement        = [1, 1]
    # signed rank agreement = [1, 1]
    feature_importance_exp1 = np.array([0.5, -0.1]).reshape((1,2))
    feature_importance_exp2 = np.array([0.3, -0.2]).reshape((1,2))
    
    feature_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'feature')
    feature_agreement_exp1_exp2 = feature_agreement_list[1]
    
    rank_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'rank')
    rank_agreement_exp1_exp2 = rank_agreement_list[1]
    
    sign_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'sign')
    sign_agreement_exp1_exp2 = sign_agreement_list[1]
    
    signed_rank_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'signed_rank')
    signed_rank_agreement_exp1_exp2 = signed_rank_agreement_list[1]

    #print(feature_agreement_list)
    
    if feature_agreement_exp1_exp2[1,0] == 1 and feature_agreement_exp1_exp2[2,0] == 1:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if rank_agreement_exp1_exp2[1,0] == 1 and rank_agreement_exp1_exp2[2,0] == 1:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if sign_agreement_exp1_exp2[1,0] == 1 and sign_agreement_exp1_exp2[2,0] == 1:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if signed_rank_agreement_exp1_exp2[1,0] == 1 and signed_rank_agreement_exp1_exp2[2,0] == 1:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    
    print('--- Fourth test ---')
    # feature agreement     = [1, 1]
    # rank agreement        = [1, 1]
    # sign agreement        = [0, 0.5]
    # signed rank agreement = [0, 0.5]
    feature_importance_exp1 = np.array([0.5, -0.1]).reshape((1,2))
    feature_importance_exp2 = np.array([-0.3, -0.2]).reshape((1,2))
    
    feature_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'feature')
    feature_agreement_exp1_exp2 = feature_agreement_list[1]
    
    rank_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'rank')
    rank_agreement_exp1_exp2 = rank_agreement_list[1]
    
    sign_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'sign')
    sign_agreement_exp1_exp2 = sign_agreement_list[1]
    
    signed_rank_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'signed_rank')
    signed_rank_agreement_exp1_exp2 = signed_rank_agreement_list[1]

    #print(feature_agreement_list)
    
    if feature_agreement_exp1_exp2[1,0] == 1 and feature_agreement_exp1_exp2[2,0] == 1:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if rank_agreement_exp1_exp2[1,0] == 1 and rank_agreement_exp1_exp2[2,0] == 1:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if sign_agreement_exp1_exp2[1,0] == 0 and sign_agreement_exp1_exp2[2,0] == 0.5:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if signed_rank_agreement_exp1_exp2[1,0] == 0 and signed_rank_agreement_exp1_exp2[2,0] == 0.5:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')


    print('--- Final test ---')
    # feature agreement     = [0.5, 1]
    # rank agreement        = [0.5, 0.5]
    # sign agreement        = [0.25, 0.625]
    # signed rank agreement = [0.25, 0.325]
    feature_importance_exp1 = np.array([[0.2, 0.1], [0.5, -0.1], [0.5, -0.1], [0.5, -0.1]])
    feature_importance_exp2 = np.array([[-0.1, -0.2], [0.2, -0.3], [0.3, -0.2], [-0.3, -0.2]])
    
    feature_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'feature')
    feature_agreement_exp1_exp2 = feature_agreement_list[1]
    
    rank_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'rank')
    rank_agreement_exp1_exp2 = rank_agreement_list[1]
    
    sign_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'sign')
    sign_agreement_exp1_exp2 = sign_agreement_list[1]
    
    signed_rank_agreement_list = get_agreement([feature_importance_exp1, feature_importance_exp2], 'signed_rank')
    signed_rank_agreement_exp1_exp2 = signed_rank_agreement_list[1]

    #print(feature_agreement_list)
    
    if feature_agreement_exp1_exp2[1,0] == 0.5 and feature_agreement_exp1_exp2[2,0] == 1:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if rank_agreement_exp1_exp2[1,0] == 0.5 and rank_agreement_exp1_exp2[2,0] == 0.5:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if sign_agreement_exp1_exp2[1,0] == 0.25 and sign_agreement_exp1_exp2[2,0] == 0.625:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    if signed_rank_agreement_exp1_exp2[1,0] == 0.25 and signed_rank_agreement_exp1_exp2[2,0] == 0.375:
        print('\033[92mTEST PASSED\033[0m')
    else:
        raise ValueError('\033[91mTEST FAILED\033[0m')
    
    print('Very good, all tests passed!')


def get_feature_importance_groundtruth(dataset_name):
    # traindata = data.AmlsimDataset(node_file='data/{dataset_name}/bank/train/nodes.csv', edge_file='data/{dataset_name}/bank/train/edges.csv', node_features=True, node_labels=True).get_data()
    # testdata = data.AmlsimDataset(node_file='data/{dataset_name}/bank/test/nodes.csv', edge_file='data/{dataset_name}/bank/test/edges.csv', node_features=True, node_labels=True).get_data()
    # traindata_U = data.AmlsimDataset(node_file='data/{dataset_name}/bank/train/nodes_U.csv', edge_file='data/{dataset_name}/bank/train/edges_U.csv', node_features=True, node_labels=True).get_data()
    # testdata_U = data.AmlsimDataset(node_file='data/{dataset_name}/bank/test/nodes_U.csv', edge_file='data/{dataset_name}/bank/test/edges_U.csv', node_features=True, node_labels=True).get_data()
    
    print('OBS BYT UT TILL ATT LADDA IN RÄTT FILER FÖR ATT SKAPA GROUNDTRUTH')
    
    traindata = SimpleNamespace()
    traindata.x = torch.tensor([[0.3, 0.2, 0.1], [0.31, 0.21, 0.11], [0.32, 0.22, 0.12]])
    
    testdata = SimpleNamespace()
    testdata.x = torch.tensor([[0.31, 0.21, 0.11], [0.32, 0.22, 0.12], [0.33, 0.23, 0.13]])
    
    traindata_U = SimpleNamespace()
    traindata_U.x = torch.tensor([[0.25, 0.15, 0.05], [0.36, 0.26, 0.16], [0.32, 0.22, 0.12]])
    
    testdata_U = SimpleNamespace()
    testdata_U.x = torch.tensor([[0.24, 0.15, 0.06], [0.35, 0.26, 0.17], [0.31, 0.22, 0.13]])
    
    train = traindata.x
    test = testdata.x
    train_U = traindata_U.x
    test_U = testdata_U.x
    
    n_samples_train = train.shape[0]
    n_samples_test = test.shape[0]
    n_features = train.shape[1]
    
    delta_train = torch.zeros((n_samples_train, n_features))
    for i in range(n_samples_train):
        delta_train[i] = train_U[i] - train[i]
    
    delta_train_mean = delta_train.mean(dim=0, keepdim=True)    
    delta_train_std = delta_train.std(dim=0, keepdim=True)

    feature_importance_groundtruth = torch.zeros((n_samples_test, n_features))
    for i in range(n_samples_test):
        delta_test_i = test_U[i] - test[i]
        feature_importance_groundtruth[i] = (delta_test_i-delta_train_mean)/delta_train_std
    
    return feature_importance_groundtruth


def correctness(model_name, dataset_name):
    print('Running correctness evalution...')
    print('OBS! KONTROLLERA ATT VI KÖR MED ALLA SAMPLES I feature_importance_list.')
    print('OBS! BYT UT feature_importance_groundtruth till att ladda in rätt data.')
    print('OBSSS MÅSTE IMPLEMENTERA DET DÄR MED DISTRIBUTION delta_i OSV FÖR ATT SKAPA groundtruth!!!')
    
    feature_importance_SVX, _, _ = load_evaluation_data_SVX(model_name, dataset_name)
    feature_importance_LIME, _ = load_evaluation_data_LIME(model_name, dataset_name)
    feature_importance_SHAP = load_evaluation_data_SHAP(model_name, dataset_name)
    
    # Använd det här för att ladda in feature_importance på riktigt
    #feature_importance_groundtruth = np.load(f'evaluation_data/feature_importance_groundtruth.npy')
    
    # OBS! Byt ut feature_importance_groundtruth till att ladda in rätt data.
    # feature_importance_groundtruth = LADDA IN DATA HÄR
    feature_importance_groundtruth = np.arange(31).reshape(1,31)
    feature_importance_groundtruth = np.repeat(feature_importance_groundtruth, 11, axis=0)
    for i in range(11):
        feature_importance_groundtruth[i] = np.random.permutation(feature_importance_groundtruth[i])
    
    importance_order = 'noabs'
    evaluation_type = 'correctness'
    
    explainer_names_1 = ['SVX', 'Groundtruth']
    explainer_names_2 = ['LIME', 'Groundtruth']
    explainer_names_3 = ['SHAP', 'Groundtruth']
    
    feature_importance_list_1 = [feature_importance_SVX[:11], feature_importance_groundtruth]
    feature_importance_list_2 = [feature_importance_LIME[:11], feature_importance_groundtruth]
    feature_importance_list_3 = [feature_importance_SHAP[:11], feature_importance_groundtruth]
    
    agreement_type = 'feature'
    feature_agreement_list_1 = get_agreement(feature_importance_list_1, agreement_type, importance_order)
    feature_agreement_list_2 = get_agreement(feature_importance_list_2, agreement_type, importance_order)
    feature_agreement_list_3 = get_agreement(feature_importance_list_3, agreement_type, importance_order)
    create_and_save_agreement_plots(feature_agreement_list_1, explainer_names_1, model_name, dataset_name, agreement_type, evaluation_type)
    create_and_save_agreement_plots(feature_agreement_list_2, explainer_names_2, model_name, dataset_name, agreement_type, evaluation_type)
    create_and_save_agreement_plots(feature_agreement_list_3, explainer_names_3, model_name, dataset_name, agreement_type, evaluation_type)
    
    agreement_type = 'rank'
    feature_agreement_list_1 = get_agreement(feature_importance_list_1, agreement_type, importance_order)
    feature_agreement_list_2 = get_agreement(feature_importance_list_2, agreement_type, importance_order)
    feature_agreement_list_3 = get_agreement(feature_importance_list_3, agreement_type, importance_order)
    create_and_save_agreement_plots(feature_agreement_list_1, explainer_names_1, model_name, dataset_name, agreement_type, evaluation_type)
    create_and_save_agreement_plots(feature_agreement_list_2, explainer_names_2, model_name, dataset_name, agreement_type, evaluation_type)
    create_and_save_agreement_plots(feature_agreement_list_3, explainer_names_3, model_name, dataset_name, agreement_type, evaluation_type)


def remove_feature(dataset, feature_index, mean_feature_values):
    dataset.x[0, feature_index] = torch.tensor(mean_feature_values[feature_index])
    return dataset


def remove_node(dataset, sar_index, node_index):
    dataset.x[node_index] = dataset.x[sar_index]
    return dataset


def completeness(model_name, dataset_name):
    print('Running completteness evaluation...')
    print('OBS: Kontrollera att vi loopar över alla sar indices.')

    evaluation_type = 'completeness'
    figures_save_dir = f'{WD_PATH}/evaluation_figures/{model_name}/{dataset_name}/{evaluation_type}'
    if not os.path.isdir(figures_save_dir):
        os.makedirs(figures_save_dir)
    data_save_dir = f'{WD_PATH}/evaluation_data/{evaluation_type}_data/{model_name}/{dataset_name}/{evaluation_type}'
    if not os.path.isdir(data_save_dir):
        os.makedirs(data_save_dir)
    
    model, traindata, testdata, feature_names, target_names = load_model(model_name, dataset_name)
    mean_feature_values = traindata.x.mean(dim=0, keepdim=True).to('cpu').detach().numpy().squeeze()

    rerun_and_save = 0
    n_sar_indices_used = 11
    
    print('NOTE')
    print(f'rerun_and_save == {rerun_and_save}')
    print(f'n_sar_indices_used == {n_sar_indices_used}')
    ans = input(f'Type "y" to continue, "n" to abort: ')
    if ans != 'y':
        raise ValueError('Aborting.')
    
    explanation_names = ['LIME', 'SHAP', 'SVX']

    for explanation_name in explanation_names:
        if explanation_name == 'LIME':
            feature_importance, _ = load_evaluation_data_LIME(model_name, dataset_name)
            node_importance = np.array([])
        elif explanation_name == 'SHAP':
            feature_importance = load_evaluation_data_SHAP(model_name, dataset_name)
            node_importance = np.array([])
        elif explanation_name == 'SVX':
            feature_importance, node_importance, _ = load_evaluation_data_SVX(model_name, dataset_name)
        else:
            raise ValueError('Explanation method not implemented.')
        
        sar_indices = get_sar_indices(model, testdata)
        n_sar_indices = len(sar_indices)
        
        print('max feature importance: ', np.max(feature_importance[:n_sar_indices_used]))
        print('min feature importance: ', np.min(feature_importance[:n_sar_indices_used]))
        print('abs_min: ', np.min(np.abs(feature_importance[:n_sar_indices_used])))
        
        # --- noabs, from negative to positive ---
        thresholds = np.arange(-1, 1.001, 0.001)
        #thresholds[1000] = 0
        thresholds = np.round(thresholds, 3)
        n_thresholds = len(thresholds)
        
        probability_diff = np.zeros((n_sar_indices, n_thresholds))
        k_count = np.zeros((n_sar_indices, n_thresholds))
        
        if rerun_and_save:
            for i, sar_index in enumerate(sar_indices[:n_sar_indices_used]):
                
                # Extract subgraph around SAR index
                if model_name == 'GAT' or model_name == 'GraphSAGE':
                    subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(sar_index.item(), 3, testdata.edge_index, relabel_nodes=False)
                    org_to_new_mapping, new_to_org_mapping, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
                    testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])
                else:
                    raise ValueError('Model not implemented.')
                
                # Calculate the original class probability
                probability_orig = model.forward(testdata_expl.x, testdata_expl.edge_index)[0][1].exp().to('cpu').detach().numpy()
                
                # For each threshold
                for gamma, thresh in enumerate(thresholds):
                    
                    # Determine which features and nodes that should be 'removed'
                    feature_index_to_remove = np.where(feature_importance[i,:] <= thresh)[0]
                    #print('node importance: ', node_importance[i][0])
                    node_index_to_remove = (np.where(node_importance[i][0] <= thresh)[0] + 1) if len(node_importance) > 0 else np.array([]) # + 1 because the 0th node in the graph is the SAR-node
                    
                    # Remove the features/nodes by replacing with mean values/SAR node feature vector
                    for feature_index in feature_index_to_remove:
                        testdata_expl = remove_feature(testdata_expl, feature_index, mean_feature_values)
                    for node_index in node_index_to_remove:
                        testdata_expl = remove_node(testdata_expl, sar_index, node_index)
                    
                    # Calculate the new class probability and the difference to the original class probability
                    probability_new = model.forward(testdata_expl.x, testdata_expl.edge_index)[0][1].exp().to('cpu').detach().numpy()
                    probability_diff[i,gamma] = probability_orig - probability_new
                    
                    # Keep track of how many features+nodes that have been removed at this threshold
                    k_count[i,gamma] = len(feature_index_to_remove) + len(node_index_to_remove)
                print('Done calculating for sar index ', i)
            
            avg_probability_diff = np.mean(probability_diff[:n_sar_indices_used], axis=0)
            std_probability_diff = np.std(probability_diff[:n_sar_indices_used], axis=0)
            avg_k_count = np.mean(k_count[:n_sar_indices_used], axis=0)

            np.save(f'{data_save_dir}/noabs_{explanation_name}_probability_diff.npy', probability_diff)    
            np.save(f'{data_save_dir}/noabs_{explanation_name}_avg_probability_diff.npy', avg_probability_diff)
            np.save(f'{data_save_dir}/noabs_{explanation_name}_std_probability_diff.npy', std_probability_diff)
            np.save(f'{data_save_dir}/noabs_{explanation_name}_k_count.npy', k_count)
            np.save(f'{data_save_dir}/noabs_{explanation_name}_avg_k_count.npy', avg_k_count)
        else:
            probability_diff = np.load(f'{data_save_dir}/noabs_{explanation_name}_probability_diff.npy')
            avg_probability_diff = np.load(f'{data_save_dir}/noabs_{explanation_name}_avg_probability_diff.npy')
            std_probability_diff = np.load(f'{data_save_dir}/noabs_{explanation_name}_std_probability_diff.npy')
            k_count = np.load(f'{data_save_dir}/noabs_{explanation_name}_k_count.npy')
            avg_k_count = np.load(f'{data_save_dir}/noabs_{explanation_name}_avg_k_count.npy')
        
        plt.figure()
        
        plt.subplot(2,2,1)
        plt.plot(thresholds,avg_probability_diff)
        plt.xlabel('Thresholds')
        plt.ylabel('AVG(y_orig - y_new)')
        plt.axvline(x=0, color='r', linestyle='--')
        
        plt.subplot(2,2,2)
        plt.plot(avg_k_count,avg_probability_diff)
        plt.xlabel('Average k')
        plt.ylabel('AVG(y_orig - y_new)')
        idx = np.where(abs(thresholds) < 0.0001)[0][0]
        plt.axvline(x=avg_k_count[idx], color='r', linestyle='--')    

        plt.subplot(2,2,3)
        plt.plot(thresholds,std_probability_diff)
        plt.xlabel('Thresholds')
        plt.ylabel('STD(y_orig - y_new)')
        plt.axvline(x=0, color='r', linestyle='--')
        
        plt.subplot(2,2,4)
        plt.plot(avg_k_count,std_probability_diff)
        plt.xlabel('Average k')
        plt.ylabel('STD(y_orig - y_new)')
        idx = np.where(abs(thresholds) < 0.0001)[0][0]
        plt.axvline(x=avg_k_count[idx], color='r', linestyle='--')
        
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()
        plt.savefig(f'{figures_save_dir}/noabs_{explanation_name}.png')

        avg_k_count_noabs = avg_k_count

        # --- noabs, from positive to negative ---
        thresholds = np.arange(1, -1.001, -0.001)
        #thresholds[1000] = 0
        thresholds = np.round(thresholds,3)
        n_thresholds = len(thresholds)
        
        probability_diff = np.zeros((n_sar_indices, n_thresholds))
        k_count = np.zeros((n_sar_indices, n_thresholds))
        
        if rerun_and_save:
            for i, sar_index in enumerate(sar_indices[:n_sar_indices_used]):
                
                # Extract subgraph around SAR index
                if model_name == 'GAT' or model_name == 'GraphSAGE':
                    subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(sar_index.item(), 3, testdata.edge_index, relabel_nodes=False)
                    org_to_new_mapping, new_to_org_mapping, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
                    testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])
                else:
                    raise ValueError('Model not implemented.')
                
                # Calculate the original class probability
                probability_orig = model.forward(testdata_expl.x, testdata_expl.edge_index)[0][1].exp().to('cpu').detach().numpy()
                
                # For each threshold
                for gamma, thresh in enumerate(thresholds):
                    
                    # Determine which features and nodes that should be 'removed'
                    feature_index_to_remove = np.where(feature_importance[i,:] > thresh)[0]
                    #print('node importance: ', node_importance[i][0])
                    node_index_to_remove = (np.where(node_importance[i][0] > thresh)[0] + 1) if len(node_importance) > 0 else np.array([]) # + 1 because the 0th node in the graph is the SAR-node
                    
                    # Remove the features/nodes by replacing with mean values/SAR node feature vector
                    for feature_index in feature_index_to_remove:
                        testdata_expl = remove_feature(testdata_expl, feature_index, mean_feature_values)
                    for node_index in node_index_to_remove:
                        testdata_expl = remove_node(testdata_expl, sar_index, node_index)
                    
                    # Calculate the new class probability and the difference to the original class probability
                    probability_new = model.forward(testdata_expl.x, testdata_expl.edge_index)[0][1].exp().to('cpu').detach().numpy()
                    probability_diff[i,gamma] = probability_orig - probability_new
                    
                    # Keep track of how many features+nodes that have been removed at this threshold
                    k_count[i,gamma] = len(feature_index_to_remove) + len(node_index_to_remove)
                print('Done calculating for sar index ', i)
            
            avg_probability_diff = np.mean(probability_diff[:n_sar_indices_used], axis=0)
            std_probability_diff = np.std(probability_diff[:n_sar_indices_used], axis=0)
            avg_k_count = np.mean(k_count[:n_sar_indices_used], axis=0)

            np.save(f'{data_save_dir}/noabs_reversed_{explanation_name}_probability_diff.npy', probability_diff)    
            np.save(f'{data_save_dir}/noabs_reversed_{explanation_name}_avg_probability_diff.npy', avg_probability_diff)
            np.save(f'{data_save_dir}/noabs_reversed_{explanation_name}_std_probability_diff.npy', std_probability_diff)
            np.save(f'{data_save_dir}/noabs_reversed_{explanation_name}_k_count.npy', k_count)
            np.save(f'{data_save_dir}/noabs_reversed_{explanation_name}_avg_k_count.npy', avg_k_count)
        else:
            probability_diff = np.load(f'{data_save_dir}/noabs_reversed_{explanation_name}_probability_diff.npy')
            avg_probability_diff = np.load(f'{data_save_dir}/noabs_reversed_{explanation_name}_avg_probability_diff.npy')
            std_probability_diff = np.load(f'{data_save_dir}/noabs_reversed_{explanation_name}_std_probability_diff.npy')
            k_count = np.load(f'{data_save_dir}/noabs_reversed_{explanation_name}_k_count.npy')
            avg_k_count = np.load(f'{data_save_dir}/noabs_reversed_{explanation_name}_avg_k_count.npy')
        
        plt.figure()
        
        plt.subplot(2,2,1)
        plt.plot(thresholds,avg_probability_diff)
        plt.xlabel('Thresholds')
        plt.ylabel('AVG(y_orig - y_new)')
        plt.axvline(x=0, color='r', linestyle='--')
        
        plt.subplot(2,2,2)
        plt.plot(avg_k_count,avg_probability_diff)
        plt.xlabel('Average k')
        plt.ylabel('AVG(y_orig - y_new)')
        idx = np.where(abs(thresholds) < 0.0001)[0][0]
        plt.axvline(x=avg_k_count[idx], color='r', linestyle='--')   
        print('idx: ', idx) 

        plt.subplot(2,2,3)
        plt.plot(thresholds,std_probability_diff)
        plt.xlabel('Thresholds')
        plt.ylabel('STD(y_orig - y_new)')
        plt.axvline(x=0, color='r', linestyle='--')
        
        plt.subplot(2,2,4)
        plt.plot(avg_k_count,std_probability_diff)
        plt.xlabel('Average k')
        plt.ylabel('STD(y_orig - y_new)')
        idx = np.where(abs(thresholds) < 0.0001)[0][0]
        plt.axvline(x=avg_k_count[idx], color='r', linestyle='--')
        print('idx: ', idx)
        
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.show()
        plt.savefig(f'{figures_save_dir}/noabs_reversed_{explanation_name}.png')
        
        avg_k_count_noabs_reversed = avg_k_count
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(range(975,1025), avg_k_count_noabs[975:1025], color = 'b')
        plt.plot(range(975,1025), avg_k_count_noabs_reversed[975:1025], color='r', linestyle='--')
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(range(975,1025), avg_k_count_noabs[975:1025], color = 'b')
        plt.plot(range(975,1025), avg_k_count_noabs_reversed[1025:975:-1], color='r', linestyle='-.')
        plt.plot(range(975,1025), avg_k_count_noabs_reversed[1025:975:-1] + avg_k_count_noabs[975:1025], color='g', linestyle=':')
        
        plt.show()
        plt.savefig(f'{figures_save_dir}/{explanation_name}_avg_k_count_comparison.png')

        # # --- abs ---
        # thresholds = np.arange(0, 1, 0.001) # Använd 0.4 ist.
        # n_thresholds = len(thresholds)
        
        # probability_diff = np.zeros((n_sar_indices, n_thresholds))
        # k_count = np.zeros((n_sar_indices, n_thresholds))
        
        # if rerun_and_save:
        #     for i, sar_index in enumerate(sar_indices[:n_sar_indices_used]):
                
        #         # Extract subgraph around SAR index
        #         if model_name == 'GAT':
        #             subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(sar_index.item(), 3, testdata.edge_index, relabel_nodes=False)
        #             org_to_new_mapping, new_to_org_mapping, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
        #             testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])
        #         else:
        #             raise ValueError('Model not implemented.')
                
        #         # Calculate the original class probability
        #         probability_orig = model.forward(testdata_expl.x, testdata_expl.edge_index)[0][1].exp().to('cpu').detach().numpy()
                
        #         # For each threshold
        #         for gamma, thresh in enumerate(thresholds):
                    
        #             # Determine which features and nodes that should be 'removed'
        #             feature_index_to_remove = np.where(np.abs(feature_importance[i,:]) < thresh)[0]
        #             node_index_to_remove = np.where(np.abs(node_importance[i][0]) < thresh)[0] if len(node_importance) > 0 else np.array([])
                    
        #             # Remove the features/nodes by replacing with mean values/SAR node feature vector
        #             for feature_index in feature_index_to_remove:
        #                 testdata_expl = remove_feature(testdata_expl, feature_index, mean_feature_values)
        #             for node_index in node_index_to_remove:
        #                 testdata_expl = remove_node(testdata_expl, sar_index, node_index)
                    
        #             # Calculate the new class probability and the difference to the original class probability
        #             probability_new = model.forward(testdata_expl.x, testdata_expl.edge_index)[0][1].exp().to('cpu').detach().numpy()
        #             probability_diff[i,gamma] = probability_orig - probability_new
                    
        #             # Keep track of how many features+nodes that have been removed at this threshold
        #             k_count[i,gamma] = len(feature_index_to_remove) + len(node_index_to_remove)
        #         print('Done calculating for sar index ', i)
            
        #     avg_probability_diff = np.mean(probability_diff[:n_sar_indices_used], axis=0)
        #     std_probability_diff = np.std(probability_diff[:n_sar_indices_used], axis=0)
        #     avg_k_count = np.mean(k_count[:n_sar_indices_used], axis=0)

        #     np.save(f'{data_save_dir}/abs_{explanation_name}_probability_diff.npy', probability_diff)    
        #     np.save(f'{data_save_dir}/abs_{explanation_name}_avg_probability_diff.npy', avg_probability_diff)
        #     np.save(f'{data_save_dir}/abs_{explanation_name}_std_probability_diff.npy', std_probability_diff)
        #     np.save(f'{data_save_dir}/abs_{explanation_name}_k_count.npy', k_count)
        #     np.save(f'{data_save_dir}/abs_{explanation_name}_avg_k_count.npy', avg_k_count)
        # else:
        #     probability_diff = np.load(f'{data_save_dir}/abs_{explanation_name}_probability_diff.npy')
        #     avg_probability_diff = np.load(f'{data_save_dir}/abs_{explanation_name}_avg_probability_diff.npy')
        #     std_probability_diff = np.load(f'{data_save_dir}/abs_{explanation_name}_std_probability_diff.npy')
        #     k_count = np.load(f'{data_save_dir}/abs_{explanation_name}_k_count.npy')
        #     avg_k_count = np.load(f'{data_save_dir}/abs_{explanation_name}_avg_k_count.npy')
        
        # plt.figure()
        
        # plt.subplot(2,2,1)
        # plt.plot(thresholds,avg_probability_diff)
        # plt.xlabel('Thresholds')
        # plt.ylabel('AVG(y_orig - y_new)')
        
        # plt.subplot(2,2,2)
        # plt.plot(avg_k_count,avg_probability_diff)
        # plt.xlabel('Average k')
        # plt.ylabel('AVG(y_orig - y_new)')

        # plt.subplot(2,2,3)
        # plt.plot(thresholds,std_probability_diff)
        # plt.xlabel('Thresholds')
        # plt.ylabel('STD(y_orig - y_new)')
        
        # plt.subplot(2,2,4)
        # plt.plot(avg_k_count,std_probability_diff)
        # plt.xlabel('Average k')
        # plt.ylabel('STD(y_orig - y_new)')
        
        # plt.subplots_adjust(wspace=0.5, hspace=0.5)
        # plt.show()
        # plt.savefig(f'{figures_save_dir}/abs_{explanation_name}.png')


def completeness2(model_name, dataset_name):
    explanation_name = 'LIME'
    n_sar_indices_used = 100
    
    thresholds = np.arange(0, 1, 0.001)
    probability_diff = np.load(f'/home/tomas/desktop/flib/thesis_XAML/completeness_data/abs_{explanation_name}_probability_diff.npy')
    k_count = np.load(f'/home/tomas/desktop/flib/thesis_XAML/completeness_data/abs_{explanation_name}_k_count.npy')
    
    def get_exceeding_threshold(probability_diff, thresholds, alpha):
        for idx, value in enumerate(probability_diff):
            if abs(value) > alpha:
                return thresholds[idx]
        return thresholds[-1]
    
    
    # Fine grained 0.001 to 0.01
    alphas = np.arange(0.001, 0.01, 0.001)
    exceeding_thresholds = list(range(len(alphas)))
    k_features_removed = list(range(len(alphas)))
    avg_exceeding_thresholds = list(range(len(alphas)))
    avg_k_features_removed = list(range(len(alphas)))
    std_exceeding_thresholds = list(range(len(alphas)))
    std_k_features_removed = list(range(len(alphas)))
    for i, alpha in enumerate(alphas):
        exceeding_thresholds_i = np.array([])
        k_features_removed_i = np.array([]) 
        for j in range(n_sar_indices_used):
            tmp = get_exceeding_threshold(probability_diff[j], thresholds, alpha)
            if tmp != thresholds[-1]:
                exceeding_thresholds_i = np.append(exceeding_thresholds_i, tmp)
                k_features_removed_i = np.append(k_features_removed_i, k_count[j, np.where(thresholds == tmp)[0][0]])
        exceeding_thresholds[i] = exceeding_thresholds_i
        k_features_removed[i] = k_features_removed_i
        avg_exceeding_thresholds[i] = np.mean(exceeding_thresholds_i)
        std_exceeding_thresholds[i] = np.std(exceeding_thresholds_i)
        avg_k_features_removed[i] = np.mean(k_features_removed_i)
        std_k_features_removed[i] = np.std(k_features_removed_i)
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.boxplot(exceeding_thresholds, labels=[f'{alpha:.3f}' for alpha in alphas])
    plt.xticks(rotation='vertical')
    plt.xlabel('Alpha')
    plt.ylabel('Exceeding Threshold')
    
    plt.subplot(2,2,3)
    plt.boxplot(k_features_removed, labels=[f'{alpha:.3f}' for alpha in alphas])
    plt.xticks(rotation='vertical')
    plt.xlabel('Alpha')
    plt.ylabel('Number of features removed')
    
    plt.subplot(2,2,2)
    plt.errorbar(alphas, avg_exceeding_thresholds, yerr=std_exceeding_thresholds, fmt='o')
    plt.xlabel('Alpha')
    plt.ylabel('Average Exceeding Threshold')
    plt.xticks(rotation='vertical')
    plt.xticks(np.arange(0.001, 0.01, 0.001))
    
    plt.subplot(2,2,4)
    plt.errorbar(alphas, avg_k_features_removed, yerr=std_k_features_removed, fmt='o')
    plt.xlabel('Alpha')
    plt.ylabel('Average number of features removed')
    plt.xticks(rotation='vertical')
    plt.xticks(np.arange(0.001, 0.01, 0.001))
    
    plt.show()
    plt.savefig('complt2_1.png')
    
    # Coarse grained 0.01 to 0.2
    alphas = np.arange(0.01, 0.2, 0.01)
    exceeding_thresholds = list(range(len(alphas)))
    k_features_removed = list(range(len(alphas)))
    avg_exceeding_thresholds = list(range(len(alphas)))
    avg_k_features_removed = list(range(len(alphas)))
    std_exceeding_thresholds = list(range(len(alphas)))
    std_k_features_removed = list(range(len(alphas)))
    for i, alpha in enumerate(alphas):
        exceeding_thresholds_i = np.array([])
        k_features_removed_i = np.array([]) 
        for j in range(n_sar_indices_used):
            tmp = get_exceeding_threshold(probability_diff[j], thresholds, alpha)
            if tmp != thresholds[-1]:
                exceeding_thresholds_i = np.append(exceeding_thresholds_i, tmp)
                k_features_removed_i = np.append(k_features_removed_i, k_count[j, np.where(thresholds == tmp)[0][0]])
        exceeding_thresholds[i] = exceeding_thresholds_i
        k_features_removed[i] = k_features_removed_i
        avg_exceeding_thresholds[i] = np.mean(exceeding_thresholds_i)
        std_exceeding_thresholds[i] = np.std(exceeding_thresholds_i)
        avg_k_features_removed[i] = np.mean(k_features_removed_i)
        std_k_features_removed[i] = np.std(k_features_removed_i)
    
    
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.boxplot(exceeding_thresholds, labels=[f'{alpha:.3f}' for alpha in alphas])
    plt.xticks(rotation='vertical')
    plt.xlabel('Alpha')
    plt.ylabel('Exceeding Threshold')
    
    plt.subplot(2,2,3)
    plt.boxplot(k_features_removed, labels=[f'{alpha:.3f}' for alpha in alphas])
    plt.xticks(rotation='vertical')
    plt.xlabel('Alpha')
    plt.ylabel('Number of features removed')
    
    plt.subplot(2,2,2)
    plt.errorbar(alphas, avg_exceeding_thresholds, yerr=std_exceeding_thresholds, fmt='o')
    plt.xlabel('Alpha')
    plt.ylabel('Average Exceeding Threshold')
    plt.xticks(rotation='vertical')
    plt.xticks(np.arange(0.01, 0.2, 0.01))
    
    
    plt.subplot(2,2,4)
    plt.errorbar(alphas, avg_k_features_removed, yerr=std_k_features_removed, fmt='o')
    plt.xlabel('Alpha')
    plt.ylabel('Average number of features removed')
    plt.xticks(rotation='vertical')
    plt.xticks(np.arange(0.01, 0.2, 0.01))
    
    plt.show()
    plt.savefig('complt2_2.png')


def confidence(model_name, dataset_name):
    print('Running confidence evaluation...')
    # Alternative 1 (High priority): Boxplots for GraphSVX and LIME showing distribution of r2 values
    _, _, r2_SVX = load_evaluation_data_SVX(model_name, dataset_name)
    _, r2_LIME = load_evaluation_data_LIME(model_name, dataset_name)

    # Boxplot for r2 values
    data = [r2_SVX[:10].squeeze(), r2_LIME[:10].squeeze()]
    labels = ['SVX', 'LIME']

    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.xlabel('Explainer')
    plt.ylabel('r2 values')
    plt.title('Comparison of r2 values between SVX and LIME')
    plt.show()
    
    evaluation_type = 'confidence'
    save_dir = f'{WD_PATH}/evaluation_figures/{model_name}/{dataset_name}/{evaluation_type}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(f'{save_dir}/r2_values.png')
    
    # Alternative 2 (Medium priority): Scatter plot with x = r2 values, y = feature agreement between GraphSVX/LIME and Groundtruth. Is there a correlation?
    # Alternative 3 (Low priority): Repeat the evaluation of each exapliner several times and look at the standard deviation of the feature importances, or calculate confidence bound and use width to quantify confidence
    

def coherence(model_name, dataset_name):
    print('Running coherence evalution...')
    print('OBS KONTROLLERA ATT VI KÖR MED ALLA SAMPLES I feature_importance_list.')
    
    feature_importance_SVX, _, _ = load_evaluation_data_SVX(model_name, dataset_name)
    feature_importance_LIME, _ = load_evaluation_data_LIME(model_name, dataset_name)
    feature_importance_SHAP = load_evaluation_data_SHAP(model_name, dataset_name)
    
    feature_importance_list = [feature_importance_SVX[:11], feature_importance_LIME[:11], feature_importance_SHAP[:11]]
    
    agreement_type = 'feature'
    evaluation_type = 'coherence'
    explainer_names = ['SVX', 'LIME', 'SHAP']
    
    for agreement_type in ['feature', 'rank', 'sign', 'signed_rank']:
        feature_agreement_list = get_agreement(feature_importance_list, agreement_type)
        create_and_save_agreement_plots(feature_agreement_list, explainer_names, model_name, dataset_name, agreement_type, evaluation_type)


def test():
    model_name = 'GAT'
    dataset_name = '100K_accts_EASY25'
    
    node_to_inspect = 3
    
    model, traindata, testdata, feature_names, target_names = load_model(model_name, dataset_name)
    
    sar_class_probas = model.forward(testdata.x, testdata.edge_index)[:,1].exp().to('cpu').detach().numpy()
    print(f'First 10 class_probas: {sar_class_probas[:10]}')
    print(f'Average probability: {np.mean(sar_class_probas)}')
    
    
    subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(node_to_inspect, 3, testdata.edge_index, relabel_nodes=False)
    org_to_new_mapping, new_to_org_mapping, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
    testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])
    
    model = model.to('cuda:0')
    testdata_expl = testdata_expl.to('cuda:0')
    model.set_test_data(testdata_expl)
    model.set_node_to_explain(0)
    
    from torch.autograd import Variable
    class_prob_fn = model.forward_NFVinput
    is_sar_class_prob_fn = lambda x: class_prob_fn( Variable( torch.from_numpy(x) ) ).detach().numpy()[:,1].reshape(-1,1)
    
    NFVinput = testdata.x[node_to_inspect]
    print(f'NFVinput = {NFVinput} | original_input = {testdata.x[node_to_inspect]}')
    
    sar_class_probas_NFVinput = is_sar_class_prob_fn(NFVinput.to('cpu').numpy().squeeze())
    print(f'NFVinput class_prob: {sar_class_probas_NFVinput} | original_input class_prob: {sar_class_probas[node_to_inspect]}')

    # --- Testa inte byta ut grafstrukturen ---
    node_to_inspect = 5
    subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(node_to_inspect, 3, testdata.edge_index, relabel_nodes=False)
    org_to_new_mapping, new_to_org_mapping, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
    testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])
    model = model.to('cuda:0')
    testdata_expl = testdata_expl.to('cuda:0')
    model.set_test_data(testdata_expl)
    model.set_node_to_explain(0)
    model.set_masking_mode('all_nodes')
    NFV_probas_nographchange = np.zeros((10,1))
    for i in range(10):
        NFVinput = testdata.x[i]
        NFV_probas_nographchange[i] = is_sar_class_prob_fn(NFVinput.to('cpu').numpy().squeeze())
    
    # --- Testa byta ut grafstrukturen också ---
    NFV_probas_graphchange = np.zeros((10,1))
    for i in range(10):
        node_to_inspect = i
        subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(node_to_inspect, 3, testdata.edge_index, relabel_nodes=False)
        org_to_new_mapping, new_to_org_mapping, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
        testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])
        model = model.to('cuda:0')
        testdata_expl = testdata_expl.to('cuda:0')
        model.set_test_data(testdata_expl)
        model.set_node_to_explain(0)
        model.set_masking_mode('all_nodes')
        NFVinput = testdata.x[i]
        NFV_probas_graphchange[i] = is_sar_class_prob_fn(NFVinput.to('cpu').numpy().squeeze())
        
    # --- Printa resultat ---
    print('Without changing graph: ')
    for i in range(10):
        print(f'NFV: {NFV_probas_nographchange[i]} | orig: {sar_class_probas[i]}')
    print(f'NFVmean: {np.mean(NFV_probas_nographchange)} | origmean: {np.mean(sar_class_probas[:10])}')
    print()
    print('Changing graph: ')
    for i in range(10):
        print(f'NFV: {NFV_probas_graphchange[i]} | orig: {sar_class_probas[i]}')
    print(f'NFVmean: {np.mean(NFV_probas_graphchange)} | origmean: {np.mean(sar_class_probas[:10])}')
    print()

    NFVavgproba_one_masked = np.zeros((10,1))
    mean_feature_values = traindata.x.mean(dim=0, keepdim=True).to('cpu').detach().numpy().squeeze()
    for i in range(10):
        node_to_inspect = i
        subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(node_to_inspect, 3, testdata.edge_index, relabel_nodes=False)
        org_to_new_mapping, new_to_org_mapping, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
        testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])
        model = model.to('cuda:0')
        testdata_expl = testdata_expl.to('cuda:0')
        model.set_test_data(testdata_expl)
        model.set_node_to_explain(0)
        model.set_masking_mode('only_node_to_explain')
        NFVavgproba_one_masked[i] = is_sar_class_prob_fn(mean_feature_values)


    NFVavgproba_all_masked = np.zeros((10,1))
    mean_feature_values = traindata.x.mean(dim=0, keepdim=True).to('cpu').detach().numpy().squeeze()
    for i in range(10):
        node_to_inspect = i
        subset_expl, edge_index_expl, _, _ = torch_geometric.utils.k_hop_subgraph(node_to_inspect, 3, testdata.edge_index, relabel_nodes=False)
        org_to_new_mapping, new_to_org_mapping, edges_new = utils.node_index_mapping(subset_expl, edge_index_expl)
        testdata_expl = Data(x = testdata.x[subset_expl], edge_index = edges_new, y = testdata.y[subset_expl])
        model = model.to('cuda:0')
        testdata_expl = testdata_expl.to('cuda:0')
        model.set_test_data(testdata_expl)
        model.set_node_to_explain(0)
        model.set_masking_mode('all_nodes')
        NFVavgproba_all_masked[i] = is_sar_class_prob_fn(mean_feature_values)
    

    print('Average probability masking all features on all nodes with mean values from traindata.')
    for i in range(10):
        print(f'NFVavgproba_one_masked node {i}: {NFVavgproba_one_masked[i]}')
    print()
    print('Average probability masking all features on all nodes with mean values from traindata.')
    for i in range(10):
        print(f'NFVavgproba_all_masked node {i}: {NFVavgproba_all_masked[i]}')


def test2():
    a = np.random.permutation(np.arange(-1,1.5,0.5)).reshape((1,-1))
    a = np.repeat(a, 3, axis=0)
    print('a: ', a)
    k = a.shape[1]
    print('k: ',k)
    ind_topk = np.argpartition(a, range(-k, 0), axis=1)[:, :-(k+1):-1]
    print('ind_topk: ', ind_topk)


def test3():
    thresholds = np.arange(1, -1.001, -0.001)
    #thresholds[1000] = 0
    thresholds = np.round(thresholds, 3)
    print(thresholds[998:1002])
    thresholds = np.arange(-1, 1.001, 0.001)
    #thresholds[1000] = 0
    thresholds = np.round(thresholds, 3)
    print(thresholds[998:1002])


def main():
    print('Running main.')
    
    model_name = 'GAT'
    dataset_name = '100K_accts_EASY25'
    
    # Calculate feature importance and save data to the folder 'evaluation_data'
    #calculate_feature_importance_LIME(model_name, dataset_name)
    #calculate_feature_importance_SHAP(model_name, dataset_name)
    #calculate_feature_importance_SVX(model_name, dataset_name)
    
    # Run the different Co12 evaluation methods
    #correctness(model_name, dataset_name)
    completeness(model_name, dataset_name)
    #confidence(model_name, dataset_name)
    #coherence(model_name, dataset_name)
    #get_feature_importance_groundtruth('hello')

    #completeness2(model_name, dataset_name)

if __name__ == "__main__":
    #test()
    #test2()
    #test3()
    main()
    
