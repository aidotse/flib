import torch
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
import optuna
from torch_geometric.loader import DataLoader

from data import AmlsimDataset
from modules import GCN, LogisticRegressor, GraphSAGE, GINe
from criterions import ClassBalancedLoss

def set_random_seed(seed:int=1):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: If you want every run to be exactly the same each time
        ##       uncomment the following lines
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_gcn(device):
    # set seed
    set_random_seed(42)
    
    # data
    traindata = AmlsimDataset(node_file='data/1bank/bank/trainset/nodes.csv', edge_file='data/1bank/bank/trainset/edges.csv').get_data()
    testdata = AmlsimDataset(node_file='data/1bank/bank/testset/nodes.csv', edge_file='data/1bank/bank/testset/edges.csv').get_data()
    traindata = traindata.to(device)
    testdata = testdata.to(device)
    
    # normalize features
    mean = traindata.x.mean(dim=0, keepdim=True)
    std = traindata.x.std(dim=0, keepdim=True)
    traindata.x = (traindata.x - mean) / std
    testdata.x = (testdata.x - mean) / std
    
    # model
    input_dim = 10
    hidden_dim = 16
    output_dim = 2
    n_layers = 3
    dropout = 0.3
    model = GCN(input_dim, hidden_dim, output_dim, n_layers, dropout)
    model.to(device)
    
    # optimizer
    lr = 0.0001
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # loss function
    beta = 0.99999999
    n_samples_per_classes = [(traindata.y == 0).sum().item(), (traindata.y == 1).sum().item()]
    criterion = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes, loss_type='sigmoid')
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(traindata)
        loss = criterion(out, traindata.y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(testdata)
                loss = criterion(out, testdata.y)
                balanced_accuracy = balanced_accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                precision = precision_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                print(f'epoch: {epoch + 1}, loss: {loss:.4f}, balanced_accuracy: {balanced_accuracy:.4f}, precision: {precision:.4f}')

def train_logistic_regressor():
    # set seed
    set_random_seed(42)
    
    # set device
    device = torch.device('cuda:0')
    
    # data
    traindata = AmlsimDataset(node_file='data/1bank/bank/trainset/nodes.csv', edge_file='data/1bank/bank/trainset/edges.csv', node_features=True, node_labels=True).get_data()
    testdata = AmlsimDataset(node_file='data/1bank/bank/testset/nodes.csv', edge_file='data/1bank/bank/testset/edges.csv', node_features=True, node_labels=True).get_data()
    traindata = traindata.to(device)
    testdata = testdata.to(device)
    
    # normalize features
    mean = traindata.x.mean(dim=0, keepdim=True)
    std = traindata.x.std(dim=0, keepdim=True)
    traindata.x = (traindata.x - mean) / std
    testdata.x = (testdata.x - mean) / std
    
    # model
    input_dim = 10
    output_dim = 2
    model = LogisticRegressor(input_dim, output_dim)
    model.to(device)
    
    # optimizer
    lr = 0.09997929137152188
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # loss function
    beta = 0.9999999994459677
    n_samples_per_classes = [(traindata.y == 0).sum().item(), (traindata.y == 1).sum().item()]
    criterion = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes, loss_type='sigmoid')
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(traindata.x)
        loss = criterion(out, traindata.y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                out = model(testdata.x)
                loss = criterion(out, testdata.y)
                accuracy = accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                balanced_accuracy = balanced_accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                precision = precision_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                recall = recall_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                f1 = f1_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                print(f'epoch: {epoch + 1}, loss: {loss:.4f}, accuracy: {accuracy:.4f}, balanced_accuracy: {balanced_accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')

def train_graph_sage():
    # set seed
    set_random_seed(42)
    
    # set device
    device = torch.device('cuda:0')
    
    # data
    traindata = AmlsimDataset(node_file='data/1bank/bank/trainset/nodes.csv', edge_file='data/1bank/bank/trainset/edges.csv', node_features=True, node_labels=True).get_data()
    testdata = AmlsimDataset(node_file='data/1bank/bank/testset/nodes.csv', edge_file='data/1bank/bank/testset/edges.csv', node_features=True, node_labels=True).get_data()
    traindata = traindata.to(device)
    testdata = testdata.to(device)
    
    # normalize features
    mean = traindata.x.mean(dim=0, keepdim=True)
    std = traindata.x.std(dim=0, keepdim=True)
    traindata.x = (traindata.x - mean) / std
    testdata.x = (testdata.x - mean) / std
    
    # create dataloader
    batch_size = 64
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False)
    
    # model
    input_dim = 10
    hidden_dim = 65
    output_dim = 2
    dropout = 0.07279450042274103
    model = GraphSAGE(input_dim, hidden_dim, output_dim, dropout)
    model.to(device)
    
    # optimizer
    lr = 0.010353064733105691
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # loss function
    beta = 0.9999999914740594
    n_samples_per_classes = [(traindata.y == 0).sum().item(), (traindata.y == 1).sum().item()]
    criterion = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes, loss_type='sigmoid')
    
    for epoch in range(100):
        set_random_seed(42+epoch+1)
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            out = model(traindata)
            loss = criterion(out, traindata.y)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    out = model(testdata)
                    loss = criterion(out, testdata.y)
                    accuracy = accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                    balanced_accuracy = balanced_accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                    precision = precision_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                    recall = recall_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                    f1 = f1_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                    print(f'epoch: {epoch + 1}, loss: {loss:.4f}, accuracy: {accuracy:.4f}, balanced_accuracy: {balanced_accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')

def train_gine():
    # set seed
    set_random_seed(42)
    
    # set device
    device = torch.device('cuda:0')
    
    # data
    traindata = AmlsimDataset(node_file='data/1bank/bank/trainset/nodes.csv', edge_file='data/1bank/bank/trainset/edges.csv', edge_features=True, edge_labels=True).get_data()
    testdata = AmlsimDataset(node_file='data/1bank/bank/testset/nodes.csv', edge_file='data/1bank/bank/testset/edges.csv', edge_features=True, edge_labels=True).get_data()
    traindata = traindata.to(device)
    testdata = testdata.to(device)
    
    # normalize node features
    mean = traindata.x.mean(dim=0, keepdim=True)
    std = traindata.x.std(dim=0, keepdim=True)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32), std)
    traindata.x = (traindata.x - mean) / std
    testdata.x = (testdata.x - mean) / std
    
    # normalize edge features
    mean = traindata.edge_attr.mean(dim=0, keepdim=True)
    std = traindata.edge_attr.std(dim=0, keepdim=True)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32), std)
    traindata.edge_attr = (traindata.edge_attr - mean) / std
    testdata.edge_attr = (testdata.edge_attr - mean) / std
    
    # model
    num_features = 1 
    num_gnn_layers = 3
    n_classes=2
    n_hidden=100
    edge_updates=True
    residual=False
    edge_dim=9
    dropout=0.0
    final_dropout=0.3140470339629592
    model = GINe(num_features, num_gnn_layers, n_classes, n_hidden, edge_updates, residual, edge_dim, dropout, final_dropout)
    model.to(device)
    
    # optimizer
    lr = 0.0401571404356884
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # loss function
    beta = 0.9999999948211576
    n_samples_per_classes = [(traindata.y == 0).sum().item(), (traindata.y == 1).sum().item()]
    criterion = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes, loss_type='sigmoid')
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(traindata)
        loss = criterion(out, traindata.y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                out = model(testdata)
                loss = criterion(out, testdata.y)
                accuracy = accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                balanced_accuracy = balanced_accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                precision = precision_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                recall = recall_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                f1 = f1_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1), zero_division=0)
                print(f'epoch: {epoch + 1}, loss: {loss:.4f}, accuracy: {accuracy:.4f}, balanced_accuracy: {balanced_accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}')

class GraphSageTrainer():
    def __init__(self, seed, device, train_node_file, train_edge_file, test_node_file, test_edge_file) -> None:
        # set seed
        self.seed = seed
        
        # set device
        self.device = device
        
        # get data
        self.traindata = AmlsimDataset(train_node_file, train_edge_file, node_features=True, node_labels=True).get_data()
        self.testdata = AmlsimDataset(test_node_file, test_edge_file, node_features=True, node_labels=True).get_data()
        self.traindata = self.traindata.to(self.device)
        self.testdata = self.testdata.to(self.device)
        
        # normalize features
        mean = self.traindata.x.mean(dim=0, keepdim=True)
        std = self.traindata.x.std(dim=0, keepdim=True)
        self.traindata.x = (self.traindata.x - mean) / std
        self.testdata.x = (self.testdata.x - mean) / std
        
        # parameters
        self.input_dim = 10
        self.output_dim = 2
    
    def objective(self, trial:optuna.Trial):
        # hyperparameters
        hidden_dim = trial.suggest_int('hidden_dim', 10, 100)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        beta = trial.suggest_float('beta', 0.99999999, 0.9999999999)
        Optimizer = getattr(optim, trial.suggest_categorical('optimizer', ['SGD', 'Adam']))
        
        # set seed
        set_random_seed(self.seed)
        
        # model
        model = GraphSAGE(self.input_dim, hidden_dim, self.output_dim, dropout)
        model.to(self.device)
        
        # optimizer
        optimizer = Optimizer(model.parameters(), lr=lr)
        
        # loss function
        n_samples_per_classes = [(self.traindata.y == 0).sum().item(), (self.traindata.y == 1).sum().item()]
        criterion = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes, loss_type='sigmoid')
        
        # train
        for epoch in range(100):
            set_random_seed(42+epoch+1)
            model.train()
            optimizer.zero_grad()
            out = model(self.traindata)
            loss = criterion(out, self.traindata.y)
            loss.backward()
            optimizer.step()

        # eval
        model.eval()
        with torch.no_grad():
            out = model(self.testdata)
            loss = criterion(out, self.testdata.y)
        
        return loss
    
    def optimize_hyperparameters(self, direction:str='minimize', n_trials:int=100):
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective, n_trials=n_trials)
        print('\nbest hyperparameters')
        for key, value in study.best_params.items():
            print(f'  {key}: {value}')
        print(f'  loss: {study.best_value}\n ')

class LogRegTrainer():
    def __init__(self, seed, device, train_node_file, train_edge_file, test_node_file, test_edge_file) -> None:
        # set seed
        self.seed = seed
        
        # set device
        self.device = device
        
        # get data
        self.traindata = AmlsimDataset(train_node_file, train_edge_file, node_features=True, node_labels=True).get_data()
        self.testdata = AmlsimDataset(test_node_file, test_edge_file, node_features=True, node_labels=True).get_data()
        self.traindata = self.traindata.to(self.device)
        self.testdata = self.testdata.to(self.device)
        
        # normalize features
        mean = self.traindata.x.mean(dim=0, keepdim=True)
        std = self.traindata.x.std(dim=0, keepdim=True)
        self.traindata.x = (self.traindata.x - mean) / std
        self.testdata.x = (self.testdata.x - mean) / std
        
        # parameters
        self.input_dim = 10
        self.output_dim = 2
    
    def objective(self, trial:optuna.Trial):
        # hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        beta = trial.suggest_float('beta', 0.99999999, 0.9999999999)
        Optimizer = getattr(optim, trial.suggest_categorical('optimizer', ['SGD', 'Adam']))
        
        # set seed
        set_random_seed(self.seed)
        
        # model
        model = LogisticRegressor(self.input_dim, self.output_dim)
        model.to(self.device)
        
        # optimizer
        optimizer = Optimizer(model.parameters(), lr=lr)
        
        # loss function
        n_samples_per_classes = [(self.traindata.y == 0).sum().item(), (self.traindata.y == 1).sum().item()]
        criterion = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes, loss_type='sigmoid')
        
        # train
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            out = model(self.traindata.x)
            loss = criterion(out, self.traindata.y)
            loss.backward()
            optimizer.step()

        # eval
        model.eval()
        with torch.no_grad():
            out = model(self.testdata.x)
            loss = criterion(out, self.testdata.y)
        
        return loss
    
    def optimize_hyperparameters(self, direction:str='minimize', n_trials:int=100):
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective, n_trials=n_trials)
        print('\nbest hyperparameters')
        for key, value in study.best_params.items():
            print(f'  {key}: {value}')
        print(f'  loss: {study.best_value}\n ')

class GINeTrainer():
    def __init__(self, seed, device, train_node_file, train_edge_file, test_node_file, test_edge_file) -> None:
        # set seed
        self.seed = seed
        
        # set device
        self.device = device
        
        # get data
        self.traindata = AmlsimDataset(train_node_file, train_edge_file, edge_features=True, edge_labels=True).get_data()
        self.testdata = AmlsimDataset(test_node_file, test_edge_file, edge_features=True, edge_labels=True).get_data()
        self.traindata = self.traindata.to(self.device)
        self.testdata = self.testdata.to(self.device)
        
        # normalize features
        mean = self.traindata.x.mean(dim=0, keepdim=True)
        std = self.traindata.x.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32), std)
        self.traindata.x = (self.traindata.x - mean) / std
        self.testdata.x = (self.testdata.x - mean) / std
        
        # parameters
        self.num_features = 1 
        self.n_classes=2
        self.edge_dim=9
        
    def objective(self, trial:optuna.Trial):
        # hyperparameters
        num_gnn_layers = trial.suggest_int('num_gnn_layers', 2, 5)
        n_hidden = trial.suggest_int('n_hidden', 10, 100)
        final_dropout = trial.suggest_float('final_dropout', 0.0, 0.5)
        edge_updates = trial.suggest_categorical('edge_updates', [True, False])
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        beta = trial.suggest_float('beta', 0.99999999, 0.9999999999)
        Optimizer = getattr(optim, trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'RMSprop']))
        
        # set seed
        set_random_seed(self.seed)
        
        # model
        model = GINe(num_features=self.num_features, num_gnn_layers=num_gnn_layers, n_classes=self.n_classes, n_hidden=n_hidden, edge_updates=edge_updates, edge_dim=self.edge_dim, final_dropout=final_dropout)
        model.to(self.device)
        
        # optimizer
        optimizer = Optimizer(model.parameters(), lr=lr)
        
        # loss function
        n_samples_per_classes = [(self.traindata.y == 0).sum().item(), (self.traindata.y == 1).sum().item()]
        criterion = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes, loss_type='sigmoid')
        
        # train
        for epoch in range(100):
            set_random_seed(42+epoch+1)
            model.train()
            optimizer.zero_grad()
            out = model(self.traindata)
            loss = criterion(out, self.traindata.y)
            loss.backward()
            optimizer.step()
        
        # eval
        model.eval()
        with torch.no_grad():
            out = model(self.testdata)
            loss = criterion(out, self.testdata.y)
        
        return loss
    
    def optimize_hyperparameters(self, direction:str='minimize', n_trials:int=100):
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective, n_trials=n_trials)
        print('\nbest hyperparameters')
        for key, value in study.best_params.items():
            print(f'  {key}: {value}')
        print(f'  loss: {study.best_value}\n ')

def main():
    #seed = 42
    #device = torch.device('cuda:0')
    #train_node_file='data/1bank/bank/trainset/nodes.csv'
    #train_edge_file='data/1bank/bank/trainset/edges.csv'
    #test_node_file='data/1bank/bank/testset/nodes.csv'
    #test_edge_file='data/1bank/bank/testset/edges.csv'
    #direction = 'minimize'
    #n_trials = 100
    #
    #trainer = GINeTrainer(seed=seed, device=device, train_node_file=train_node_file, train_edge_file=train_edge_file, test_node_file=test_node_file, test_edge_file=test_edge_file)
    #trainer.optimize_hyperparameters(direction=direction, n_trials=n_trials)
    
    print('training logreg')
    train_logistic_regressor()
    print()
    
    print('training graphsage')
    train_graph_sage()
    print()
    
    print('training gine')
    train_gine()
    print()

if __name__ == "__main__":
    main()