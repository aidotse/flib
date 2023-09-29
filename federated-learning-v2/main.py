import logging
import copy 
import torch
import pandas as pd
from data import BankDataset
from client import Client
from server import Server
from modules import LogisticRegressor
import datetime
import os

logging.basicConfig(filename='log', encoding='utf-8', level=logging.INFO)

def run_experiment(name, n_workers, trainsets, testsets, Module, Optimizer, Criterion, lr, n_epochs, batch_size, n_rounds, eval_every, n_rounds_no_aggregation):
    
    clients = []
    for i, trainset in enumerate(trainsets):
        clients.append(Client(
            name=f'client_{i}',
            device=torch.device('cpu'),
            trainset=trainset,
            valset=None, 
            testset=copy.deepcopy(testsets[0]), 
            Module=Module, 
            Optimizer=Optimizer, 
            Criterion=Criterion, 
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size
        ))   
    input_dim = len(trainsets[0].columns) - 1
    output_dim = len(trainsets[0][trainsets[0].columns[-1]].unique())
    module = LogisticRegressor(input_dim=input_dim, output_dim=output_dim)
    model = module.state_dict()
    server = Server(clients, model, n_workers)
    
    server.run(n_rounds=n_rounds, eval_every=eval_every, n_rounds_no_aggregation=n_rounds_no_aggregation)
    
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.makedirs(f'results/{dt}', exist_ok=True)
    os.system(f'mv log results/{dt}/log')

def tune_hyperparameters():
    pass

def main():
    
    # hyperparameters
    n_rounds = 101
    eval_every = 10
    n_rounds_no_aggregation = 101
    Module = LogisticRegressor 
    Optimizer = torch.optim.SGD
    Criterion = torch.nn.CrossEntropyLoss
    lr = 0.02
    n_epochs = 1 
    batch_size = 64
    n_workers = 1
    
    # load data
    DATASET = '10K_accts_super_easy'
    path = f'datasets/{DATASET}'
    trainsets, _, testsets = BankDataset(path).datasets()
    #trainsets = [pd.concat(trainsets).reset_index(drop=True)] # uncoment for centralized learning
     
    # init clients
    clients = []
    for i, trainset in enumerate(trainsets):
        clients.append(Client(
            name=f'client_{i}',
            device=torch.device('cpu'),
            trainset=trainset,
            valset=None, 
            testset=copy.deepcopy(testsets[0]), 
            Module=Module, 
            Optimizer=Optimizer, 
            Criterion=Criterion, 
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size
        ))
    
    # init server
    input_dim = len(trainsets[0].columns) - 1
    output_dim = len(trainsets[0][trainsets[0].columns[-1]].unique())
    module = LogisticRegressor(input_dim=input_dim, output_dim=output_dim)
    model = module.state_dict()
    server = Server(clients, model, n_workers)
    
    # train
    server.run(n_rounds=n_rounds, eval_every=eval_every, n_rounds_no_aggregation=n_rounds_no_aggregation)
    
    # save results
    dt = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    os.makedirs(f'results/{dt}', exist_ok=True)
    os.system(f'mv log results/{dt}/log')

if __name__ == '__main__':
    main()
