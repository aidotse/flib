from importlib import reload
import numpy as np
import torch
import torch_geometric
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, precision_score, recall_score, fbeta_score, precision_recall_curve, average_precision_score
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GINEConv, GATConv, BatchNorm, Linear
from torch_geometric.data import Data
import copy
import optuna
import optuna.visualization.matplotlib as opt_viz
import os
import time

import criterions
reload(criterions)
from criterions import ClassBalancedLoss
import data
reload(data)
from torch_geometric.data import Data

import train
reload(train)
from train import train_GAT_GraphSVX_foroptuna, train_GraphSAGE_foroptuna

import modules
reload(modules)

from utils import EarlyStopper


def train_loop_GAT(dataset_name, hyperparameters = None, verbose = False):
    # Computing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('Double check which dataset is being used.')
    
    # Data
    node_file_train = f'ultra_final_data/{dataset_name}/bank/train/nodes.csv'
    edge_file_train = f'ultra_final_data/{dataset_name}/bank/train/edges.csv'
    node_file_test = f'ultra_final_data/{dataset_name}/bank/test/nodes.csv'
    edge_file_test = f'ultra_final_data/{dataset_name}/bank/test/edges.csv'  
    
    traindata = data.AmlsimDataset(node_file=node_file_train, edge_file=edge_file_train, node_features=True, node_labels=True).get_data()
    testdata = data.AmlsimDataset(node_file=node_file_test, edge_file=edge_file_test, node_features=True, node_labels=True).get_data()
    traindata = torch_geometric.transforms.ToUndirected()(traindata)
    testdata = torch_geometric.transforms.ToUndirected()(testdata)
    feature_names = ['sum','mean','median','std','max','min','in_degree','out_degree','n_unique_in','n_unique_out']
    target_names = ['not_sar','is_sar']
    
    # Normalization (note: normalizing also integer features | maybe add other preprocessing)
    mean = traindata.x.mean(dim=0, keepdim=True)
    std = traindata.x.std(dim=0, keepdim=True)
    traindata.x = (traindata.x - mean) / std
    testdata.x = (testdata.x - mean) / std
    
    traindata = traindata.to(device)
    testdata = testdata.to(device)

    # Non-tunable hyperparameters
    in_channels = traindata.x.shape[1]
    out_channels = 2
    
    # Tunable hyperparamters
    hidden_channels = hyperparameters['hidden_channels']
    num_heads = hyperparameters['num_heads']
    dropout = hyperparameters['dropout']
    lr = hyperparameters['lr']
    epochs = hyperparameters['epochs']
    beta = hyperparameters['beta']
        
    # Model
    model = modules.GAT(in_channels=in_channels,
                        hidden_channels=hidden_channels,
                        out_channels=out_channels,
                        num_heads=num_heads,
                        dropout=dropout)
    
    model = model.to(device)
    model.set_return_type('logits')
    #model.set_return_type('log_probas')
    
    # Criterion and optimizer
    n_samples_per_classes_train = [(traindata.y == 0).sum().item(), (traindata.y == 1).sum().item()]
    n_samples_per_classes_test = [(testdata.y == 0).sum().item(), (testdata.y == 1).sum().item()]
    print(f'number of samples per classes (train) = {n_samples_per_classes_train}')
    print(f'number of samples per classes (test) = {n_samples_per_classes_test}')
    criterion = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes_train, loss_type='sigmoid')
    #criterion_val = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes_val, loss_type='sigmoid')
    #criterion = torch.nn.NLLLoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize early stopper
    early_stopper = EarlyStopper(patience=100, min_delta=0) #Stops after 10 epochs without improvement
    
    # Train model
    print(f'Starting training with {epochs} epochs.')
    running_train_loss = []
    running_val_loss = []
    
    # This model needs special input format, so we can't use the train_model function
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model.forward(traindata.x, traindata.edge_index)
        loss = criterion(out, traindata.y)
        running_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model.forward(testdata.x, testdata.edge_index)
            loss = criterion(out, testdata.y)
            running_val_loss.append(loss.item())
            out = F.softmax(out, dim=1)
            accuracy = accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
            precision = precision_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
            recall = recall_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
            fbeta = fbeta_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), beta=1, zero_division=0)
        if verbose and ((epoch+1)%10 == 0 or epoch == epochs-1):
                print(f'epoch: {epoch + 1}, train_loss: {running_train_loss[-1]:.4f}, val_loss: {running_val_loss[-1]:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, fbeta: {fbeta:.4f}')
        # if early_stopper.early_stop(loss):
        #     print(f'Stopping training early at {epoch}/{epochs} epochs.')             
        #     break
    print('Finished training.')
    
    return model, traindata, testdata, feature_names, target_names, accuracy, fbeta


def run_GAT_training(dataset_name):
    # hyperparameters = {'hidden_channels': 10,
    #                     'num_heads': 2,
    #                     'dropout': 0.15,
    #                     'lr': 0.01, #0.005
    #                     'epochs': 1000,
    #                     'beta': 0.99991,
    #                     'seed':1}
    hyperparameters = {'hidden_channels': 10,
                        'num_heads': 2, #Note: num_heads is hardcoded to 1 in modules
                        'dropout': 0.2,
                        'lr': 0.002,
                        'epochs': 500,
                        'beta': 0.99991,
                        'seed':1}

    model, traindata, testdata, feature_names, target_names, accuracy, fbeta = train_loop_GAT(dataset_name, hyperparameters, verbose = True)
    return model, hyperparameters, traindata, testdata, feature_names, target_names, accuracy, fbeta


def run_SAGE_training():    
    hyperparameters = {'hidden_channels': 15,
                        'num_heads': 3,
                        'dropout': 0.2,
                        'lr': 0.002,
                        'epochs': 1000,
                        'beta': 0.9999, 
                        'seed':42}

    model_name = 'GraphSAGE'
    dataset_name ='100K_accts_MID5'

    model, traindata, valdata, feature_names, target_names, running_train_loss, running_test_loss, accuracy = train_GraphSAGE_foroptuna(dataset_name, hyperparameters, verbose = True)
    return model, traindata, valdata, feature_names, target_names, running_train_loss, running_test_loss, accuracy

    checkpoint = torch.load(model_path)
    model_state_dict = checkpoint['model_state_dict']
    hyperparameters = checkpoint['hyperparameters']
    traindata = checkpoint['traindata']
    testdata = checkpoint['testdata']
    feature_names = ['sums', 'means', 'medians', 'stds', 'maxs', 'mins', 'in_sum', 'out_sum', 'in_mean', 'out_mean', 'in_median', 'out_median', 'in_std', 'out_std', 'in_max', 'out_max', 'in_min', 'out_min', 'count_in', 'count_out', 'count_days_in_bank', 'count_phone_changes', 'sums_spending', 'means_spending', 'medians_spending', 'stds_spending', 'maxs_spending', 'mins_spending', 'counts_spending']
    target_names = checkpoint['target_names']


def evaluate_performance(model, testdata, model_name, dataset_name):
    model.eval()
    with torch.no_grad():
        out = model.forward(testdata.x, testdata.edge_index)
        out = F.softmax(out, dim=1)
        y_pred = out.cpu().numpy().argmax(axis=1)
        y_true = testdata.y.cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
    
    FPpercent = cm[0,1] / (cm[1,1] + cm[0,1])
    print(FPpercent)

    # Todo: precision recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, out[:,1].cpu().numpy())
    average_precision = average_precision_score(y_true, out[:,1].cpu().numpy())

    print(f'Average Precision = {average_precision}')

    # print(thresholds)
    # print(len(thresholds))

    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    plt.savefig(f'GNN_training_figures/{model_name}_{dataset_name}_precision_recall_curve.png')


def save_model(model, hyperparameters, traindata, testdata, feature_names, target_names, model_name, dataset_name):
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': hyperparameters,
        'traindata': traindata,
        'testdata': testdata,
        'feature_names': feature_names,
        'target_names': target_names,
    }, f'trained_models/{model_name}_{dataset_name}.pth')


def load_model(model_name, dataset_name):
    
    # Load model, data, and other variables
    model_path = f'/home/agnes/desktop/flib/thesis_XAML/trained_models/{model_name}_{dataset_name}.pth'
    
    checkpoint = torch.load(model_path)
    model_state_dict = checkpoint['model_state_dict']
    hyperparameters = checkpoint['hyperparameters']
    traindata = checkpoint['traindata']
    testdata = checkpoint['testdata']
    feature_names = ['sums', 'means', 'medians', 'stds', 'maxs', 'mins', 'in_sum', 'out_sum', 'in_mean', 'out_mean', 'in_median', 'out_median', 'in_std', 'out_std', 'in_max', 'out_max', 'in_min', 'out_min', 'count_in', 'count_out', 'count_days_in_bank', 'count_phone_changes', 'sums_spending', 'means_spending', 'medians_spending', 'stds_spending', 'maxs_spending', 'mins_spending', 'counts_spending']
    target_names = checkpoint['target_names']

    # Initialize model
    if model_name == 'GAT':
        in_channels = traindata.x.shape[1]
        hidden_channels = hyperparameters['hidden_channels']
        out_channels = 2
        num_heads = hyperparameters['num_heads']
        dropout = hyperparameters['dropout']
        model = modules.GAT(in_channels = in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_heads=num_heads, dropout=dropout)
    elif model_name == 'GraphSAGE':
        in_channels = traindata.x.shape[1]
        hidden_channels = hyperparameters['hidden_channels']
        out_channels = 2
        dropout = hyperparameters['dropout']
        seed = hyperparameters['seed']
        torch.manual_seed(seed)
        print('seed: ', seed)
        model = modules.GraphSAGE_GraphSVX_foroptuna(in_channels = in_channels, hidden_channels=hidden_channels, out_channels=out_channels, dropout=dropout, seed = seed)
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
    #model.set_return_type('log_probas')

    return model, traindata, testdata, feature_names, target_names


def run_GAT_optuna(model_name, dataset_name, n_trials):
    
    def objective(trial):
        hyperparameters = {'hidden_channels': trial.suggest_int('hidden_channels', 10, 15, step=1),
                            'num_heads': 2,
                            'dropout': trial.suggest_float('dropout', 0.1, 0.3),
                            'lr': trial.suggest_float('lr', 0.001, 0.005),
                            'epochs': 500,
                            'beta': trial.suggest_float('beta', 0.9999, 0.99992)}
        
        model, traindata, testdata, feature_names, target_names, accuracy, fbeta = train_loop_GAT(dataset_name, hyperparameters, verbose = False)
        
        if trial.number == 0:
            save_model(model, hyperparameters, traindata, testdata, feature_names, target_names, model_name, dataset_name)
            print('First save.')
        elif fbeta < trial.study.best_value:
            save_model(model, hyperparameters, traindata, testdata, feature_names, target_names, model_name, dataset_name)
            print('Saved new best model.')
        
        return fbeta

    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = n_trials)
    
    output_dir = f'GNN_training_figures/optuna/{model_name}/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    # Generate and save each Optuna visualization plo
    plt.figure()
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(f'{output_dir}/optimization_history.png')
    plt.close()
    plt.figure()
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(f'{output_dir}/param_importances.png')
    plt.close()
    plt.figure()
    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig(f'{output_dir}/slice.png')
    plt.close()


def unpack_attention_output(attention_output):
    edge_index, attention_weights = attention_output
    return edge_index, attention_weights


def main():
    model_name = 'GAT'
    dataset_name = '100K_accts_MID5'
    n_trials = 50
    
    model, hyperparameters, traindata, testdata, feature_names, target_names, accuracy, fbeta = run_GAT_training(dataset_name)
    # #model, traindata, testdata, feature_names, target_names, running_train_loss, running_test_loss, accuracy = run_SAGE_training()
    
    # evaluate_performance(model, testdata, model_name, dataset_name)
    save_model(model = model,
               hyperparameters = hyperparameters,
               traindata = traindata,
               testdata = testdata,
               feature_names = feature_names,
               target_names = target_names,
               model_name = model_name,
               dataset_name = dataset_name)
    
    # time_start = time.perf_counter()
    # run_GAT_optuna(model_name, dataset_name, n_trials)
    # time_stop = time.perf_counter()
    # print(f'Running {n_trials} took {time_stop - time_start} seconds.')
    
    model, traindata, testdata, feature_names, target_names = load_model(model_name, dataset_name)
    evaluate_performance(model, testdata, model_name, dataset_name)
    
    # model.set_return_attention_weights(True)
    # x, attention_output = model.forward(testdata.x, testdata.edge_index)
    # print(len(attention_output))
    # print(attention_output[0].shape, attention_output[1].shape)

if __name__ == '__main__':
    main()
    #test()























# --- Old stuff ---

# class GAT(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout=0.3):
#         super(GAT, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.dropout = dropout
#         self.fc_pre = Linear(in_channels, hidden_channels)
#         self.gat_single = GATConv(hidden_channels, hidden_channels, heads=1, concat = False, dropout=dropout)
#         self.fc_post = Linear(hidden_channels, out_channels)
#         self.log_softmax = torch.nn.LogSoftmax(dim=1)
#         self.return_type = 'logits'
#         self.return_attention_weights = False

#     def forward(self, x, edge_index):
#         x = self.fc_pre(x)
#         x = F.dropout(x, p = self.dropout, training=self.training)
#         x = F.elu(x)
#         x, attention_weights = self.gat_single(x, edge_index, return_attention_weights=True)
#         x = F.dropout(x, p = self.dropout, training=self.training)
#         x = F.elu(x)
#         x = self.fc_post(x)
        
#         if self.return_type == 'log_probas':
#             x = self.log_softmax(x) #<--- log probas should only be used when running the explainers.
        
#         if self.return_attention_weights:
#             return x, attention_weights
#         else:
#             return x

#     def set_return_type(self, return_type):
#         if return_type == 'logits' or return_type == 'log_probas':
#             self.return_type = return_type
#         else:
#             raise ValueError('return_type must be either "logits" or "log_probas"')

#     def set_return_attention_weights(self, return_attention_weights):
#         if type(return_attention_weights) == bool:
#             self.return_attention_weights = return_attention_weights
#         else:
#             raise ValueError('return_attention_weights must be a boolean')