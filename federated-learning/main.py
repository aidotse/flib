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
import time
from sklearn.model_selection import ParameterGrid
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor as Pool
from criterions import ClassBalancedLoss
import random

#logging.basicConfig(filename='log', encoding='utf-8', level=logging.INFO)

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

def run_experiment(n_workers, trainsets, testsets, Module, Optimizer, Criterion, lr, n_epochs, batch_size, n_rounds, eval_every, n_rounds_no_aggregation, optimizer_params=None, criterion_params=None, name=None, log_predictions=False):
    
    set_random_seed(42)
    
    os.makedirs(f'results/{name}', exist_ok=True)
    
    # init clients
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
            optimizer_params=optimizer_params,
            criterion_params=criterion_params,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size
        ))
    
    # init server
    input_dim = len(trainsets[0].columns) - 1
    output_dim = len(trainsets[0][trainsets[0].columns[-1]].unique())
    module = Module(input_dim=input_dim, output_dim=output_dim)
    model = module.state_dict()
    server = Server(clients=clients, model=model, n_workers=n_workers, log_predictions=log_predictions, log_file=f'results/{name}/log')
    
    # train
    print(f'running experiment: {name}')
    avg_losses = server.run(n_rounds=n_rounds, eval_every=eval_every, n_rounds_no_aggregation=n_rounds_no_aggregation)
    print()
    
    return avg_losses[-1]

def run_experiment_mp(args):
    n_workers, trainsets, testsets, Module, Optimizer, Criterion, lr, n_epochs, batch_size, n_rounds, eval_every, n_rounds_no_aggregation, optimizer_params, criterion_params, name, log_predictions = args
    
    set_random_seed(42)
    
    os.makedirs(f'results/{name}', exist_ok=True)
    
    # init clients
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
            optimizer_params=optimizer_params,
            criterion_params=criterion_params,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size
        ))
    
    # init server
    input_dim = len(trainsets[0].columns) - 1
    output_dim = len(trainsets[0][trainsets[0].columns[-1]].unique())
    module = Module(input_dim=input_dim, output_dim=output_dim)
    model = module.state_dict()
    server = Server(clients=clients, model=model, n_workers=n_workers, log_predictions=log_predictions, log_file=f'results/{name}/log')
    
    # train
    print(f'running experiment: {name}')
    avg_losses = server.run(n_rounds=n_rounds, eval_every=eval_every, n_rounds_no_aggregation=n_rounds_no_aggregation)
    print()
    
    return avg_losses[-1]

def sweep_hyperparameters(n_workers, trainsets, testsets, Module, Optimizer, Criterion, n_epochs, batch_size, n_rounds, eval_every, n_rounds_no_aggregation, name, log_predictions=False):
    
    if os.path.exists(f'results/{name}'):
        os.system(f'rm -rf results/{name}')
    
    pg = ParameterGrid({
        'lr': [0.1, 0.2, 0.3, 0.4], #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'criterion_beta': [0.1], #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        'momentum': [0.1],
        'dampening': [0.1],
        'weight_decay': [0.1],
        
    })
    
    best_loss = np.inf
    best_params = None
    
    for i, params in enumerate(pg):
        
        avg_loss = run_experiment(
            n_workers=n_workers,
            trainsets=trainsets,
            testsets=testsets,
            Module=Module,
            Optimizer=Optimizer,
            optimizer_params={'momentum': params['momentum'], 'dampening': params['dampening'], 'weight_decay': params['weight_decay']},
            Criterion=Criterion,
            criterion_params={'beta': params['criterion_beta'], 'loss_type': 'sigmoid'},
            lr=params['lr'],
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_rounds=n_rounds,
            eval_every=eval_every,
            n_rounds_no_aggregation=n_rounds_no_aggregation,
            name=name+f'/run_{i}',
            log_predictions=log_predictions
        )
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = params
            best_name = name+f'/run_{i}'
    
    with open(f'results/{name}/best_run.txt', 'w') as f:
        f.write(f'name: {best_name}\n')
        f.write(f'loss: {best_loss}\n')
        f.write(f'params: {best_params}\n')
    
    return best_name, best_loss, best_params

def sweep_hyperparameters_mp(n_workers, trainsets, testsets, Module, Optimizer, Criterion, n_epochs, batch_size, n_rounds, eval_every, n_rounds_no_aggregation, name, log_predictions=False, pg=None):
        
        if os.path.exists(f'results/{name}'):
            os.system(f'rm -rf results/{name}')
        
        if pg is None:
            pg = ParameterGrid({
                'lr':  [0.5, 0.6, 0.7, 1.0, 1.1, 1.2, 1.3],  # [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], 
                'criterion_beta':  [0.9999, 0.99999],  # [0.9, 0.99, 0.999, 0.9999], 
                'momentum':  [0.0, 0.9], # [0.0, 0.5, 0.7, 0.9],
                'dampening':  [0.0, 0.1], # [0.0, 0.3, 0.7, 1.0],
                'weight_decay':  [0.0], # [0.0, 0.01, 0.1],
            })
        
        best_loss = np.inf
        best_params = None
        best_name = None
        
        args = []
        for i, params in enumerate(pg):
            args.append((
                n_workers,
                trainsets,
                testsets,
                Module,
                Optimizer,
                Criterion,
                params['lr'],
                n_epochs,
                batch_size,
                n_rounds,
                eval_every,
                n_rounds_no_aggregation,
                {'momentum': params['momentum'], 'dampening': params['dampening'], 'weight_decay': params['weight_decay']},
                {'beta': params['criterion_beta'], 'loss_type': 'sigmoid'},
                name+f'/run_{i}',
                log_predictions
            ))
        
        with Pool(2) as p:
            results = p.map(run_experiment_mp, args)
        
        for i, result in enumerate(results):
            if result < best_loss:
                best_loss = result
                best_params = pg[i]
                best_name = name+f'/run_{i}'
        
        with open(f'results/{name}/best_run.txt', 'w') as f:
            f.write(f'name: {best_name}\n')
            f.write(f'loss: {best_loss}\n')
            f.write(f'params: {best_params}\n')
        
        return best_name, best_loss, best_params

def main():
    
    set_random_seed(1)
    
    mp.set_start_method('spawn')
    
    print()
    
    # hyperparameters
    log_predictions = True
    n_rounds = 301
    eval_every = 30
    n_rounds_no_aggregation = 0
    Module = LogisticRegressor 
    Optimizer = torch.optim.SGD
    Criterion = ClassBalancedLoss #torch.nn.CrossEntropyLoss
    n_epochs = 1 
    batch_size = 128
    n_workers = 4
    
    # load data
    DATASET = '50K_accts'
    path = f'../datasets/{DATASET}/preprocessed/'
    trainsets, _, testsets = BankDataset(path).datasets()
    
    '''
    run_experiment(
        n_workers=n_workers, 
        trainsets=trainsets, 
        testsets=testsets, 
        Module=Module, 
        Optimizer=Optimizer, 
        Criterion=Criterion, 
        lr=1.5, 
        n_epochs=n_epochs, 
        batch_size=batch_size, 
        n_rounds=n_rounds, 
        eval_every=eval_every, 
        n_rounds_no_aggregation=0, 
        optimizer_params={'momentum': 0.9, 'dampening': 0.0, 'weight_decay': 0.0}, 
        criterion_params={'beta': 0.99999999999, 'loss_type': 'sigmoid'}
    )
    '''
    
    pg = ParameterGrid({
        'lr':  [1.4, 1.5, 1.6], 
        'criterion_beta':  [0.9999999999, 0.99999999999, 0.999999999999],  
        'momentum':  [0.9], 
        'dampening':  [0.0], 
        'weight_decay':  [0.0]
    })
    
    t = time.time()
    
    name, loss, params = sweep_hyperparameters_mp(
        n_workers=n_workers,
        trainsets=trainsets,
        testsets=testsets,
        Module=Module,
        Optimizer=Optimizer,
        Criterion=Criterion,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_rounds=n_rounds,
        eval_every=eval_every,
        n_rounds_no_aggregation=n_rounds_no_aggregation,
        name='param_sweep_fed',
        log_predictions=log_predictions,
        pg=pg
    )
    
    print()
    print('federated:')
    print(f'    run: {name}')
    print(f'    loss: {loss}')
    print(f'    params: {params}')
    print()
    
    pg = ParameterGrid({
        'lr':  [0.9, 1.0, 1.1], 
        'criterion_beta':  [0.999999999, 0.9999999999, 0.99999999999],  
        'momentum':  [0.0], 
        'dampening':  [0.0], 
        'weight_decay':  [0.0]
    })
    
    name, loss, params = sweep_hyperparameters_mp(
        n_workers=n_workers,
        trainsets=trainsets,
        testsets=testsets,
        Module=Module,
        Optimizer=Optimizer,
        Criterion=Criterion,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_rounds=n_rounds,
        eval_every=eval_every,
        n_rounds_no_aggregation=n_rounds,
        name='param_sweep_iso',
        log_predictions=log_predictions,
        pg=pg
    )
    
    print()
    print('isolation:')
    print(f'    run: {name}')
    print(f'    loss: {loss}')
    print(f'    params: {params}')
    print()
    
    pg = ParameterGrid({
        'lr':  [0.8], 
        'criterion_beta':  [0.99999999999], 
        'momentum':  [0.9], 
        'dampening':  [0.0], 
        'weight_decay':  [0.0]
    })
    
    name, loss, params = sweep_hyperparameters_mp(
        n_workers=1,
        trainsets=[pd.concat(trainsets).reset_index(drop=True)],
        testsets=testsets,
        Module=Module,
        Optimizer=Optimizer,
        Criterion=Criterion,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_rounds=n_rounds,
        eval_every=eval_every,
        n_rounds_no_aggregation=n_rounds,
        name='param_sweep_cen',
        log_predictions=log_predictions,
        pg=pg
    )
    
    print()
    print('centralized:')
    print(f'    run: {name}')
    print(f'    loss: {loss}')
    print(f'    params: {params}')
    print()
    
    print(f'time: {time.time()-t}')
    
if __name__ == '__main__':
    main()
