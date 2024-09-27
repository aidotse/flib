from flib.train import Clients
from flib.train.servers import Server
from flib.utils import set_random_seed
import multiprocessing as mp
import torch
import os
import pickle
import optuna
import time

def federated(seed=42, train_dfs=None, val_dfs=None, test_dfs=None, client='LogRegClient', criterion='ClassBalancedLoss', n_workers=3, n_rounds=100, eval_every=10, **kwargs):
    
    set_random_seed(seed)
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print(f'Start method already set to {mp.get_start_method()}')
        pass
    
    Client = getattr(Clients, client)
    
    # init clients
    clients = []
    for i, train_df, val_df, test_df in zip(range(len(train_dfs)), train_dfs, val_dfs, test_dfs):
        client = Client(
            name=f'c{i}',
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            criterion=criterion,
            **kwargs
        )
        clients.append(client)
    
    # init server
    server = Server(
        clients=clients,
        n_workers=n_workers
    )
    
    # run
    results, state_dict, avg_loss = server.run(n_rounds=n_rounds, eval_every=eval_every)
    
    return results, state_dict, avg_loss

class HyperparamTuner():
    def __init__(self, seed=42, trainsets=None, n_rounds=50, model='LogisticRegressor', optimizer=['SGD'], criterion=['ClassBalancedLoss'], beta=(0.9999, 0.99999999), local_epochs=[1], batch_size=[64, 128, 256], lr=(0.001, 0.1), n_workers=3, device='cuda:0', storage=None, results_file=None):
        self.seed = seed
        self.trainsets = trainsets
        self.n_rounds = n_rounds
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.beta = beta
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_workers = n_workers
        self.device = device
        self.storage = storage
        self.results_file = results_file
    
    def objective(self, trial):
        _, _, avg_loss = federated(
            seed=self.seed,
            trainsets=self.trainsets,
            n_rounds=self.n_rounds,
            eval_every=None,
            model=self.model,
            optimizer=trial.suggest_categorical('optimizer', self.optimizer),
            criterion=trial.suggest_categorical('criterion', self.criterion),
            beta=trial.suggest_float('beta', self.beta[0], self.beta[1]),
            local_epochs=trial.suggest_categorical('local_epochs', self.local_epochs),
            batch_size=trial.suggest_categorical('batch_size', self.batch_size),
            lr=trial.suggest_float('lr', self.lr[0], self.lr[1]),
            n_workers=self.n_workers,
            device=self.device
        )
        
        return avg_loss
    
    def optimize(self, n_trials=10):
        study = optuna.create_study(storage=self.storage, sampler=optuna.samplers.TPESampler(), study_name='study', directions=['minimize'], load_if_exists=True)
        study.optimize(self.objective, n_trials=n_trials)
        with open(self.results_file, 'a') as f:
            f.write(f'\n\n{time.ctime()}\n')
            f.write(f'seed: {self.seed}\n')
            f.write(f'trainsets: {self.trainsets}\n')
            f.write(f'n_rounds: {self.n_rounds}\n')
            f.write(f'model: {self.model}\n')
            f.write(f'optimizer: {self.optimizer}\n')
            f.write(f'criterion: {self.criterion}\n')
            f.write(f'beta: {self.beta}\n')
            f.write(f'local_epochs: {self.local_epochs}\n')
            f.write(f'batch_size: {self.batch_size}\n')
            f.write(f'lr: {self.lr}\n')
            f.write(f'n_workers: {self.n_workers}\n')
            f.write(f'device: {self.device}\n')
            f.write(f'storage: {self.storage}\n')
            f.write(f'results_file: {self.results_file}\n\n')
            f.write(f'Best hyperparameters: {study.best_params}\n')
            f.write(f'Best loss: {study.best_value}\n')
        return study.best_params, study.best_value
    