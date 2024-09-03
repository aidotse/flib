
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import data
import modules
from data import AmlsimDataset
from modules import GraphSAGE, GCN, GAT, GAT2, GAT2NO_NORM, GCN2, GraphSAGE2, CustomGAT
import torch.optim as optim
import pandas as pd
import os
import time

#Import configurations for different datasets and models
#from model_configs import configs  
from model_configs import configs  



# Reload your modules if they have been updated
reload(data)
reload(modules)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


dataset_names = ["100K_accts_EASY25_NEW_NEW","100K_accts_MID5_NEW_NEW","100K_accts_HARD1_NEW_NEW"]  # List of dataset names as they are named in your base path

ap_scores = []
results = []
validation_runs = 5

NOISE_LEVELS = ['no_noise','0.1', '0.25']

NODE_TYPES = [
      'nodes.csv',
      'nodes_missing_label.csv',
      'nodes_bipartite.csv',
      'nodes_cycle.csv',
      'nodes_fan_in.csv',
      'nodes_fan_out.csv',
      'nodes_FN.csv',
      'nodes_FP.csv',
      'nodes_gather_scatter.csv',
      'nodes_neighbour.csv',
      'nodes_scatter_gather.csv',
      'nodes_stack.csv',
]

def preprocess_data(file_path):
    """Load the CSV file, fill missing values with 0, and return a DataFrame."""
    df = pd.read_csv(file_path)
    if 'bank' in df.columns:
        df = df.drop(columns=['bank'])  # Drop the 'bank' column if it exists
    if 'true_label' in df.columns:
        df = df.drop(columns=['true_label'])
    if 'account' in df.columns:
        df = df.drop(columns=['account'])
    df.fillna(0, inplace=True)  # Fill missing values with 0
    return df

def save_preprocessed_data(df, file_path):
    """Save the preprocessed DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

# Load and normalize data
def load_and_normalize_data(dataset_name, node_type,noise_level):
    base_path = 'base path for data' # Insert base path for data include easy, mid and hard dataset
    
    # Defining file paths for noised nodes and unchanged edges
    train_node_file = f'{base_path}{dataset_name}/bank/train/{noise_level}/{node_type}'
    processed_train_node_file = train_node_file.replace('.csv', '_processed.csv')  # Define processed file path
    train_edge_file = f'{base_path}{dataset_name}/bank/train/edges.csv'
    test_node_file = f'{base_path}{dataset_name}/bank/test/nodes.csv'
    test_edge_file = f'{base_path}{dataset_name}/bank/test/edges.csv'

    # Check if the train node file exists before preprocessing
    if not os.path.exists(train_node_file):
        print(f"Node type '{node_type}' not present in noise level '{noise_level}'")
        return None, None  

    # Preprocess nodes and edges
    train_nodes = preprocess_data(train_node_file)
    train_edges = preprocess_data(train_edge_file)
    test_nodes = preprocess_data(test_node_file)
    test_edges = preprocess_data(test_edge_file)

    # Save preprocessed data back to CSV files
    save_preprocessed_data(train_nodes, processed_train_node_file)  # Save processed train nodes
    save_preprocessed_data(train_edges, train_edge_file)
    save_preprocessed_data(test_nodes, test_node_file)
    save_preprocessed_data(test_edges, test_edge_file)

    # Load processed data into AmlsimDataset instances and move to device
    # Ensure to load from the processed train node file
    traindata = AmlsimDataset(node_file=processed_train_node_file, edge_file=train_edge_file, node_features=True, node_labels=True).get_data().to(device)
    testdata = AmlsimDataset(node_file=test_node_file, edge_file=test_edge_file, node_features=True, node_labels=True).get_data().to(device)

    # Normalize features
    mean = traindata.x.mean(dim=0, keepdim=True)
    std = traindata.x.std(dim=0, keepdim=True)
    traindata.x = (traindata.x - mean) / std
    testdata.x = (testdata.x - mean) / std

    return traindata, testdata



def train_and_evaluate(config, traindata, testdata):
    
    model_class = config["model_class"]
    
    if  model_class == GAT2NO_NORM:  # GAT specific configuration
        model = model_class(config["in_channels"], config["out_channels"], config["num_heads"], config["dropout"]).to(device)
    elif  model_class == CustomGAT:  # GAT specific configuration
        model = model_class(config["in_channels"], config["out_channels"], config["num_heads"], config["dropout"]).to(device)
    elif model_class == GCN2:  # GCN specific configuration
        model = model_class(config["input_dim"], config["output_dim"],config["dropout"]).to(device)
    elif model_class == GraphSAGE2:  # GraphSAGE specific configuration
        model = model_class(config["input_dim"], config["hidden_dim"], config["output_dim"], config["dropout"]).to(device)
    else:
        raise ValueError("Unsupported model class provided in configuration")
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    valid_label_mask = traindata.y != -1  # Assuming -1 is used for missing labels

    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        out = model(traindata)
        out = out[valid_label_mask]  # Filter outputs where labels exist
        labels = traindata.y[valid_label_mask]  # Filter labels to avoid missing labels

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:  # Evaluation
            model.eval()
            with torch.no_grad():
                out = model(testdata)
                precision = precision_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                recall = recall_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                print(f"Epoch: {epoch+1}, Precision: {precision}, Recall: {recall}")

    # Final evaluation and plot
    AP, AUC = evaluate_and_plot(model, testdata)

    return AP, AUC

def evaluate_and_plot(model, testdata):
    model.eval()
    with torch.no_grad():
        out = model(testdata)
        y_scores = F.softmax(out, dim=1).cpu().numpy()[:, 1]  # Probabilities for the positive class
        y_true = testdata.y.cpu().numpy()  # True labels

    average_precision = average_precision_score(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)  # Compute AUC score

    return average_precision, auc_score


start_time = time.time()


for dataset_name in dataset_names:
    for noise_level in NOISE_LEVELS:
        for node_type in NODE_TYPES:
            traindata, testdata = load_and_normalize_data(dataset_name, node_type, noise_level)

            if traindata is None or testdata is None:
                print(f"Skipping training for {dataset_name} with node type {node_type} at noise level {noise_level}.")
                continue  # Skip to the next loop iteration if data is not available


            model_configs = configs[dataset_name]
            for model_name, config in model_configs.items():
                # Update configuration with actual input dimensions from data
                if 'input_dim' in config:
                    config['input_dim'] = traindata.num_node_features
                elif 'in_channels' in config:  # For GAT model
                    config['in_channels'] = traindata.num_node_features

                ap_scores, auc_scores = [], []
                
                for _ in range(validation_runs):
                    ap, auc = train_and_evaluate(config, traindata, testdata)
                    ap_scores.append(ap)
                    auc_scores.append(auc)

                # Calculate stats
                ap_mean = np.mean(ap_scores)
                ap_std = np.std(ap_scores)
                auc_mean = np.mean(auc_scores)
                auc_std = np.std(auc_scores)

                # Append results
                results.append({
                    'Model': model_name,
                    'Dataset': f"{dataset_name}_{noise_level}_{node_type}",
                    'Average Precision Mean': ap_mean,
                    'Average Precision Std': ap_std,
                    'AUC Score Mean': auc_mean,
                    'AUC Score Std': auc_std
                })

                # Print stats
                print(f"Stats for {model_name} on {dataset_name} with {node_type} at {noise_level}:")
                print(f"  AP Mean: {ap_mean:.4f}, AP Std: {ap_std:.4f}")
                print(f"  AUC Mean: {auc_mean:.4f}, AUC Std: {auc_std:.4f}")

                # Save results after each model configuration
                results_df = pd.DataFrame(results)
                results_csv_path = "results_summary_intermediate_NEW_bipartite.csv"
                results_df.to_csv(results_csv_path, index=False)
                print(f"Intermediate results saved to {results_csv_path}")

# Save results
results_df = pd.DataFrame(results)
results_csv_path = "results_summary_FINAL_NEW_bipartite.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Results summary saved to {results_csv_path}")

elapsed_time = time.time() - start_time
print(f"Total runtime: {elapsed_time:.2f} seconds")
