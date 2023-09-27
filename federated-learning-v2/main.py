import logging
import copy 
import torch
from data import BankDataset
from client import Client
from server import Server
from modules import LogisticRegressor
import time
import os

def main():
    
    # load data
    DATASET = '10K_accts_super_easy_prepro2'
    path = f'/home/edvin/Desktop/flib/federated-learning/datasets/{DATASET}'
    trainsets, _, testsets = BankDataset(path).datasets()
    
    # init clients
    Module = LogisticRegressor 
    Optimizer = torch.optim.SGD
    Criterion = torch.nn.CrossEntropyLoss
    lr = 0.01
    n_epochs = 1 
    batch_size = 64
    clients = []
    for i, trainset in enumerate(trainsets):
        clients.append(Client(
            name=f'client_{i}',
            device=torch.device('cuda:0'),
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
    n_workers = 4
    module = LogisticRegressor(input_dim=23, output_dim=2)
    model = module.state_dict()
    server = Server(clients, model, n_workers)
    
    # train
    n_rounds = 201
    eval_every = 10
    start = time.time()
    server.run(n_rounds=n_rounds, eval_every=eval_every)
    end = time.time()
    print()
    print(f' elapsed time: {end-start}')


if __name__ == '__main__':
    main()
