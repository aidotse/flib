from importlib import reload

import torch
import torch_geometric
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score, precision_recall_curve, balanced_accuracy_score

import data
reload(data)

import modules
reload(modules)

from criterions import ClassBalancedLoss
from utils import EarlyStopper

def train_model(model, train, test, optimizer, criterion, epochs = 200, verbose = False):
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model.forward(train)
        loss = criterion(out, train.y)
        loss.backward()
        optimizer.step()
        if verbose and ((epoch+1)%10 == 0 or epoch == epochs-1):
            model.eval()
            with torch.no_grad():
                out = model.forward(test)
                loss = criterion(out, test.y)
                precision = precision_score(test.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                recall = recall_score(test.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                print(f'epoch: {epoch + 1}, loss: {loss:.4f}, precision: {precision:.4f}, recall: {recall:.4f}')
    return model

def train_GCN(hyperparameters = None, verbose = False):
    # Computing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('Double check which dataset is being used.')
    
    # Data

    #traindata = data.AmlsimDataset(node_file='bank/train/nodes.csv', edge_file='bank/train/edges.csv', node_features=True, node_labels=True).get_data()
    #testdata = data.AmlsimDataset(node_file='bank/test/nodes.csv', edge_file='bank/test/edges.csv', node_features=True, node_labels=True).get_data()

    traindata = data.AmlsimDataset(node_file='data/simulation2/swedbank/train/nodes.csv', edge_file='data/simulation2/swedbank/train/edges.csv', node_features=True, node_labels=True).get_data()
    testdata = data.AmlsimDataset(node_file='data/simulation2/swedbank/test/nodes.csv', edge_file='data/simulation2/swedbank/test/edges.csv', node_features=True, node_labels=True).get_data()
    feature_names = ['sum','mean','median','std','max','min','in_degree','out_degree','n_unique_in','n_unique_out']
    target_names = ['not_sar','is_sar']
    
    # --- Add preprocessing here ---
    
    traindata = traindata.to(device)
    testdata = testdata.to(device)
    
    # Non-tunable hyperparameters
    input_dim = traindata.x.shape[1]
    output_dim = 2
    
    # Tunable hyperparamters
    hidden_dim = 10
    num_layers = 3
    dropout = 0.3
    lr = 0.0001
    epochs = 200
    class_weights = [1, 1.5]
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Model
    model = modules.GCN(input_dim = input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        num_layers=num_layers,
                        dropout=dropout)
    
    model = model.to(device)
    
    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train model
    print(f'Starting training with {epochs} epochs.')
    train_model(model, traindata, testdata, optimizer, criterion, epochs = epochs, verbose = verbose)
    print(f'Finished training.')
    
    return model, traindata, testdata, feature_names, target_names

def train_GCN_GraphSVX(hyperparameters = None, verbose = False):
    # Computing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('Double check which dataset is being used.')
    
    # Data
    traindata = data.AmlsimDataset(node_file='data/simulation3/swedbank/train/nodes.csv', edge_file='data/simulation3/swedbank/train/edges.csv', node_features=True, node_labels=True).get_data()
    testdata = data.AmlsimDataset(node_file='data/simulation3/swedbank/test/nodes.csv', edge_file='data/simulation3/swedbank/test/edges.csv', node_features=True, node_labels=True).get_data()
    feature_names = ['sum','mean','median','std','max','min','in_degree','out_degree','n_unique_in','n_unique_out']
    target_names = ['not_sar','is_sar']
    
    # --- Add preprocessing here ---
    
    traindata = traindata.to(device)
    testdata = testdata.to(device)
    
    # Non-tunable hyperparameters
    input_dim = traindata.x.shape[1]
    output_dim = 2
    
    # Tunable hyperparamters
    hidden_dim = 10
    num_layers = 3
    dropout = 0.3
    lr = 0.005
    epochs = 500
    class_weights = [1, 3.75]
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Model
    model = modules.GCN_GraphSVX(input_dim = input_dim,
                                 hidden_dim=hidden_dim,
                                 output_dim=output_dim,
                                 num_layers=num_layers,
                                 dropout=dropout)
    
    model = model.to(device)
    
    # Criterion and optimizer
    criterion = torch.nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train model
    print(f'Starting training with {epochs} epochs.')
    # This model needs special input format, so we can't use the train_model function
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model.forward(traindata.x, traindata.edge_index)
        loss = criterion(out, traindata.y)
        loss.backward()
        optimizer.step()
        if verbose and ((epoch+1)%10 == 0 or epoch == epochs-1):
            model.eval()
            with torch.no_grad():
                out = model.forward(testdata.x, testdata.edge_index)
                loss = criterion(out, testdata.y)
                precision = precision_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                recall = recall_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                print(f'epoch: {epoch + 1}, loss: {loss:.4f}, precision: {precision:.4f}, recall: {recall:.4f}')
    print('Finished training.')
    
    return model, traindata, testdata, feature_names, target_names

def train_GAT_GraphSVX(hyperparameters = None, verbose = False):
    # Computing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('Double check which dataset is being used.')
    # Data
    # traindata = data.AmlsimDataset(node_file='data/simulation3/swedbank/train/nodes.csv', edge_file='data/simulation3/swedbank/train/edges.csv', node_features=True, node_labels=True).get_data()
    # testdata = data.AmlsimDataset(node_file='data/simulation3/swedbank/test/nodes.csv', edge_file='data/simulation3/swedbank/test/edges.csv', node_features=True, node_labels=True).get_data()
  
    traindata = data.AmlsimDataset(node_file='bank/train/nodes.csv', edge_file='bank/train/edges.csv', node_features=True, node_labels=True).get_data()
    testdata = data.AmlsimDataset(node_file='bank/test/nodes.csv', edge_file='bank/test/edges.csv', node_features=True, node_labels=True).get_data()
    traindata = torch_geometric.transforms.ToUndirected()(traindata)
    testdata = torch_geometric.transforms.ToUndirected()(testdata)
    #feature_names = ['sum','mean','median','std','max','min','in_degree','out_degree','n_unique_in','n_unique_out']
    #feature names for:  banks, in_sum, out_sum, in_mean, out_mean, in_median, out_median, in_std, out_std, in_max, out_max, in_min, out_min, n_unique_in, n_unique_out, count_days_in_bank, count_phone_changes, sums_spending, means_spending, medians_spending, stds_spending, maxs_spending, mins_spending, counts_spending
    feature_names = ['in_sum', 'out_sum', 'in_mean', 'out_mean', 'in_median', 'out_median', 'in_std', 'out_std', 'in_max', 'out_max', 'in_min', 'out_min', 'n_unique_in', 'n_unique_out', 'count_days_in_bank', 'count_phone_changes', 'sums_spending', 'means_spending', 'medians_spending', 'stds_spending', 'maxs_spending', 'mins_spending', 'counts_spending']
    target_names = ['not_sar','is_sar']
    
    


    print('traindata', traindata)
    # --- Add preprocessing here ---
    
    traindata = traindata.to(device)
    testdata = testdata.to(device)
    
    # Non-tunable hyperparameters
    in_channels = traindata.x.shape[1]
    out_channels = 2
    
    # Tunable hyperparamters
    hidden_channels = 10
    num_heads = 3
    dropout = 0.3
    lr = 0.0001
    print('lr',lr)
    epochs = 400
    class_weights = [0.9489, 1.0511]
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Model
    model = modules.GAT_GraphSVX(in_channels=in_channels,
                                 hidden_channels=hidden_channels,
                                 out_channels=out_channels,
                                 num_heads=num_heads,
                                 dropout=dropout)
    
    model = model.to(device)
    
    # Criterion and optimizer
    criterion = torch.nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train model
    print(f'Starting training with {epochs} epochs.')
    # This model needs special input format, so we can't use the train_model function
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Get the parameters of the model
        # params = list(model.parameters())
        # print('print params', params)
        # print('printing traindata x', traindata.x[:5,:])
        # print('printing traindata edge_index', traindata.edge_index[:5,:])
        out = model.forward(traindata.x, traindata.edge_index)
        # print('printing out', out[:5,:])
        loss = criterion(out, traindata.y)
        loss.backward()
        optimizer.step()
        if verbose and ((epoch+1)%10 == 0 or epoch == epochs-1 or epoch<10):
            model.eval()
            with torch.no_grad():
                train_loss = criterion(out, traindata.y)

                out = model.forward(testdata.x, testdata.edge_index)
                test_loss = criterion(out, testdata.y)
                balanced_accuracy = balanced_accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                precision = precision_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                recall = recall_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                accuracy= accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                print(f'epoch: {epoch + 1}, test_loss: {test_loss:.4f}, train_loss: {train_loss:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, balanced_accuracy: {balanced_accuracy:.4f}, accuracy: {accuracy:.4f}')
    print('Finished training.')
    
    return model, traindata, testdata, feature_names, target_names


def train_GAT_GraphSVX_foroptuna(hyperparameters = None, verbose = False):
    # Computing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('Double check which dataset is being used.')
    
    # Data
    traindata = data.AmlsimDataset(node_file='data/100K_accts_EASY25/bank/train/nodes.csv', edge_file='data/100K_accts_EASY25/bank/train/edges.csv', node_features=True, node_labels=True).get_data()
    testdata = data.AmlsimDataset(node_file='data/100K_accts_EASY25/bank/test/nodes.csv', edge_file='data/100K_accts_EASY25/bank/test/edges.csv', node_features=True, node_labels=True).get_data()
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
    model = modules.GAT_GraphSVX_foroptuna(in_channels=in_channels,
                                            hidden_channels=hidden_channels,
                                            out_channels=out_channels,
                                            num_heads=num_heads,
                                            dropout=dropout)
    
    model = model.to(device)
    
    # Criterion and optimizer
    n_samples_per_classes_train = [(traindata.y == 0).sum().item(), (traindata.y == 1).sum().item()]
    n_samples_per_classes_test = [(testdata.y == 0).sum().item(), (testdata.y == 1).sum().item()]
    print(f'number of samples per classes (train) = {n_samples_per_classes_train}')
    print(f'number of samples per classes (test) = {n_samples_per_classes_test}')
    criterion_train = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes_train, loss_type='sigmoid')
    criterion_test = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes_test, loss_type='sigmoid')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize early stopper
    early_stopper = EarlyStopper(patience=10, min_delta=0) #Stops after 10 epochs without improvement
    
    # Train model
    print(f'Starting training with {epochs} epochs.')
    running_train_loss = []
    running_test_loss = []
    
    # This model needs special input format, so we can't use the train_model function
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model.forward(traindata.x, traindata.edge_index)
        loss = criterion_train(out, traindata.y)
        running_train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model.forward(testdata.x, testdata.edge_index)
            loss = criterion_test(out, testdata.y)
            running_test_loss.append(loss.item())
            out = F.softmax(out, dim=1)
            accuracy = accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
            precision = precision_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
            recall = recall_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
            fbeta = fbeta_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), beta=beta, zero_division=0)
        if verbose and ((epoch+1)%10 == 0 or epoch == epochs-1):
                print(f'epoch: {epoch + 1}, train_loss: {running_train_loss[-1]:.4f}, test_loss: {running_test_loss[-1]:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, fbeta: {fbeta:.4f}')
        if early_stopper.early_stop(loss):
            print(f'Stopping training early at {epoch}/{epochs} epochs.')             
            break
    print('Finished training.')
    
    return model, traindata, testdata, feature_names, target_names, running_train_loss, running_test_loss, accuracy




def train_graphSAGE_foroptuna(hyperparameters = None, verbose = False):
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # data
    traindata = data.AmlsimDataset(node_file='data/100K_accts_MID5/bank/train/nodes.csv', edge_file='data/100K_accts_MID5/bank/train/edges.csv', node_features=True, node_labels=True).get_data()
    testdata = data.AmlsimDataset(node_file='data/100K_accts_MID5/bank/test/nodes.csv', edge_file='data/100K_accts_MID5/bank/test/edges.csv', node_features=True, node_labels=True).get_data()

    traindata = traindata.to(device)
    testdata = testdata.to(device)
    
    # normalize features
    mean = traindata.x.mean(dim=0, keepdim=True)
    std = traindata.x.std(dim=0, keepdim=True)
    traindata.x = (traindata.x - mean) / std
    testdata.x = (testdata.x - mean) / std

    feature_names=[]
    target_names=[]

    # Non-tunable hyperparameters
    in_channels = traindata.x.shape[1]
    out_channels = 2

    # Tunable hyperparamters
    hidden_channels = hyperparameters['hidden_channels']
    dropout = hyperparameters['dropout']
    lr = hyperparameters['lr']
    epochs = hyperparameters['epochs']
    beta = hyperparameters['beta']
        
    # model
    model = modules.GraphSAGE_GraphSVX_foroptuna(in_channels, hidden_channels, out_channels, dropout)
    model.to(device)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Criterion and optimizer
    # n_samples_per_classes_train = [(traindata.y == 0).sum().item(), (traindata.y == 1).sum().item()]
    # n_samples_per_classes_test = [(testdata.y == 0).sum().item(), (testdata.y == 1).sum().item()]
    # print(f'number of samples per classes (train) = {n_samples_per_classes_train}')
    # print(f'number of samples per classes (test) = {n_samples_per_classes_test}')
    # criterion_train = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes_train, loss_type='sigmoid')
    # criterion_test = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes_test, loss_type='sigmoid')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    ############
    # loss function
    n_samples_per_classes = [(traindata.y == 0).sum().item(), (traindata.y == 1).sum().item()]
    criterion = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes, loss_type='sigmoid')
    running_train_loss = []
    running_test_loss = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(traindata.x, traindata.edge_index)
        loss = criterion(out, traindata.y)
        
        running_train_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                out = model.forward(testdata.x, testdata.edge_index)
                loss = criterion(out, testdata.y)
                running_test_loss.append(loss.item())
                accuracy = accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                balanced_accuracy = balanced_accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                precision = precision_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                recall = recall_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                fbeta = fbeta_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), beta=beta, zero_division=0)
                print(f'epoch: {epoch + 1}, train_loss: {running_train_loss[-1]:.4f}, test_loss: {running_test_loss[-1]}, accuracy: {accuracy:.4f}, balanced_accuracy: {balanced_accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f{beta}: {fbeta:.4f}')

    return model, traindata, testdata, feature_names, target_names, running_train_loss, running_train_loss, accuracy



