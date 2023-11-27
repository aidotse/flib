import torch
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix

from data import AmlsimDataset
from modules import GCN
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
    n_layers = 2
    dropout = 0.3
    model = GCN(input_dim, hidden_dim, output_dim, n_layers, dropout)
    model.to(device)
    
    # optimizer
    lr = 0.1
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # loss function
    beta = 0.99999999
    n_samples_per_classes = [(traindata.y == 0).sum().item(), (traindata.y == 1).sum().item()]
    criterion = ClassBalancedLoss(beta=beta, n_samples_per_classes=n_samples_per_classes, loss_type='sigmoid')
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        out = model(traindata)
        loss = criterion(out, traindata.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(testdata)
                loss = criterion(out, testdata.y)
                balanced_accuracy = balanced_accuracy_score(testdata.y.cpu().numpy(), out.cpu().numpy().argmax(axis=1))
                print(f'epoch: {epoch}, loss: {loss:.4f}, balanced_accuracy: {balanced_accuracy:.4f}')

def main():
    set_random_seed(42)
    train_gcn(torch.device('cuda:0'))

if __name__ == "__main__":
    main()