import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import os
from importlib import reload
import data
import modules
from data import AmlsimDataset
from modules import GraphSAGE, GCN, GAT, GAT2, GraphSAGE2, GCN2, GAT2NO_NORM, CustomGAT
import torch.optim as optim
import optuna

# Reload your modules if they have been updated
reload(data)
reload(modules)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

dataset_names = ["100K_accts_EASY25_NEW_NEW", "100K_accts_MID5_NEW_NEW", "100K_accts_HARD1_NEW_NEW"]

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    irrelevant_columns = ['bank', 'true_label', 'account']
    df.drop(columns=[col for col in irrelevant_columns if col in df.columns], inplace=True)
    df.fillna(0, inplace=True)
    return df

def save_preprocessed_data(df, file_path):
    """Save the preprocessed DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

# Load and normalize data
def load_and_normalize_data(dataset_name):
    base_path = '/home/edgelab/UnreliableLabels/flib/gnn/data/'
    train_node_file = f'{base_path}{dataset_name}/bank/train/no_noise/nodes.csv'
    train_edge_file = f'{base_path}{dataset_name}/bank/train/edges.csv'
    test_node_file = f'{base_path}{dataset_name}/bank/test/nodes.csv'
    test_edge_file = f'{base_path}{dataset_name}/bank/test/edges.csv'

    # Preprocess nodes and edges
    train_nodes = preprocess_data(train_node_file)
    train_edges = preprocess_data(train_edge_file)
    test_nodes = preprocess_data(test_node_file)
    test_edges = preprocess_data(test_edge_file)

    # Save preprocessed data back to CSV files
    save_preprocessed_data(train_nodes, train_node_file)
    save_preprocessed_data(train_edges, train_edge_file)
    save_preprocessed_data(test_nodes, test_node_file)
    save_preprocessed_data(test_edges, test_edge_file)

    traindata = AmlsimDataset(node_file=train_node_file, edge_file=train_edge_file, node_features=True, node_labels=True).get_data()
    testdata = AmlsimDataset(node_file=test_node_file, edge_file=test_edge_file, node_features=True, node_labels=True).get_data()
    traindata, testdata = traindata.to(device), testdata.to(device)

    mean = traindata.x.mean(dim=0, keepdim=True)
    std = traindata.x.std(dim=0, keepdim=True)
    traindata.x = (traindata.x - mean) / std
    testdata.x = (testdata.x - mean) / std

    return traindata, testdata

def objective(trial, traindata, testdata, model_class):
    # Model-specific configurations
    if model_class == GAT:
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        hidden_channels = trial.suggest_int('hidden_channels', 16, 128, step=32)
        model = GAT(traindata.num_node_features, hidden_channels, 2, num_heads, dropout).to(device)
    elif model_class == GCN:
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
        model = GCN(traindata.num_node_features, hidden_dim, 2, dropout).to(device)
    elif model_class == GCN2:
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        model = GCN2(traindata.num_node_features, 2, dropout).to(device)
    elif model_class == GAT2NO_NORM:
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        model = GAT2NO_NORM(traindata.num_node_features, 2, num_heads, dropout).to(device)
    elif model_class == CustomGAT:
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        model = CustomGAT(traindata.num_node_features, 2, num_heads, dropout).to(device)
    elif model_class == GAT2:
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        model = GAT2(traindata.num_node_features, 2, num_heads, dropout).to(device)
    elif model_class == GraphSAGE2:
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
        model = GraphSAGE2(traindata.num_node_features, hidden_dim, 2, dropout).to(device)
    
    else:  # Default to GraphSAGE
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
        model = GraphSAGE(traindata.num_node_features, hidden_dim, 2, dropout).to(device)
    

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        out = model(traindata)
        loss = criterion(out, traindata.y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
            out = model(testdata)
            y_scores = F.softmax(out, dim=1)[:, 1]  # Probabilities for class 1
            y_true = testdata.y.cpu().numpy()
            y_scores = y_scores.cpu().numpy()
            average_precision = average_precision_score(y_true, y_scores, pos_label=1)

    return average_precision

def main():
    all_results = []
    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")
        traindata, testdata = load_and_normalize_data(dataset_name)
        
        model_classes = { 'CustomGAT': CustomGAT} #'GraphSAGE': GraphSAGE, 'GCN': GCN, 
        results = {}
        for model_name, model_class in model_classes.items():
            print(f"Optimizing {model_name}")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, traindata, testdata, model_class), n_trials=30)
            results[model_name] = study.best_trial

            # Collecting best trial data
            best_trial = {
                'Model': model_name,
                'Dataset': dataset_name,
                'F1 Score': study.best_trial.value,
                'Parameters': study.best_trial.params
            }
            all_results.append(best_trial)


        # Print or save the results for each dataset and model
        for model_name, trial in results.items():
            print(f"Best AP for {model_name} on {dataset_name}: {trial.value:.4f} with params {trial.params}")
    # Convert the list of results to ah DataFrame and save to CSV
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join("results", "optuna_best_trials_summary.csv")
    os.makedirs("results", exist_ok=True)  # Ensure the results directory exists
    results_df.to_csv(results_csv_path, index=False)
    print(f"Optuna best trials summary saved to {results_csv_path}")

if __name__ == "__main__":
    main()
