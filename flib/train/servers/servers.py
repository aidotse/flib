import multiprocessing as mp
import time
import torch
from collections import OrderedDict
import copy
import numpy as np
from tqdm import tqdm
import time
import os
import pandas as pd
from flib.utils import decrease_lr

class Server():
    def __init__(self, clients, n_workers):
        self.clients = clients
        self.n_workers = n_workers
        
    def _train_clients(self, clients):
        state_dicts = []
        client_names = []
        losses = []
        tpfptnfns = []
        for client in clients:
            loss, tpfptnfn = client.train()
            client.train()
            state_dicts.append(client.get_state_dict())
            client_names.append(client.name)
            losses.append(loss)
            tpfptnfns.append(tpfptnfn)
        return client_names, losses, tpfptnfns, state_dicts
    
    def _evaluate_clients(self, clients, dataset):
        client_names = []
        losses = []
        tpfptnfns = []
        for client in clients:
            loss, tpfptnfn = client.evaluate(dataset=dataset)
            client_names.append(client.name)
            losses.append(loss)
            tpfptnfns.append(tpfptnfn)
        return client_names, losses, tpfptnfns
            
    def _average_state_dicts(self, state_dicts:OrderedDict, weights:list=None):
        if weights:
            weights = [weight/sum(weights) for weight in weights]
        else:
            weights = [1.0/len(state_dicts) for _ in state_dicts]
        with torch.no_grad():
            avg_state_dict = copy.deepcopy(state_dicts[0])
            for key in avg_state_dict:
                avg = torch.zeros_like(avg_state_dict[key])
                for state_dict, weight in zip(state_dicts, weights):
                    avg += state_dict[key] * weight
                avg_state_dict[key] = avg
        return avg_state_dict

    def run(self, n_rounds=100, eval_every=10, state_dict=None, n_no_aggregation_rounds=0, lr_patience=5, es_patience=15, **kwargs):
        
        results_dict = {client.name: {0: {}} for client in self.clients}
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        avg_state_dict = None

        with mp.Pool(self.n_workers) as p:
            
            client_splits = np.array_split(self.clients, self.n_workers)
            
            # sync state_dicts over clients
            if state_dict is None:
                state_dict = self.clients[0].get_state_dict()
            for client in self.clients[1:]:
                client.load_state_dict(copy.deepcopy(state_dict))
            
            # evaluate initial model
            results = p.starmap(self._evaluate_clients, [(client_split, 'trainset') for client_split in client_splits])
            previous_train_loss = 0.0
            for result in results:
                for client, loss, tpfptnfn in zip(result[0], result[1], result[2]):
                    results_dict[client][0]['train'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
                    previous_train_loss += loss / len(self.clients)
            if eval_every is not None:
                results = p.starmap(self._evaluate_clients, [(client_split, 'valset') for client_split in client_splits])
                previous_val_loss = 0.0
                for result in results:
                    for client, loss, tpfptnfn in zip(result[0], result[1], result[2]):
                        results_dict[client][0]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
                        previous_val_loss += loss / len(self.clients)
            
            for round in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
                
                results = p.map(self._train_clients, client_splits)
                state_dicts = []
                avg_loss = 0.0
                for result in results:
                    for client, loss, tpfptnfn, state_dict in zip(result[0], result[1], result[2], result[3]):
                        results_dict[client][round] = {'train': {'loss': loss, 'tpfptnfn': tpfptnfn}}
                        state_dicts.append(state_dict)
                        avg_loss += loss / len(self.clients)
                if avg_loss >= previous_train_loss - 0.0005:
                    lr_patience -= 1
                else:
                    lr_patience = lr_patience_reset
                if lr_patience <= 0:
                    tqdm.write('Decreasing learning rate.')
                    for client in self.clients:
                        decrease_lr(client.optimizer, factor=0.5)
                previous_train_loss = avg_loss
                
                if round > n_no_aggregation_rounds:
                    avg_state_dict = self._average_state_dicts(state_dicts)
                    for client in self.clients:
                        client.load_state_dict(copy.deepcopy(avg_state_dict))
                
                if eval_every is not None and round % eval_every == 0:
                    results = p.starmap(self._evaluate_clients, [(client_split, 'valset') for client_split in client_splits])
                    avg_loss = 0.0
                    for result in results:
                        for client, loss, tpfptnfn in zip(result[0], result[1], result[2]):
                            results_dict[client][round]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
                            avg_loss = loss / len(self.clients)
                    if avg_loss >= previous_val_loss - 0.0005:
                        es_patience -= eval_every
                    else:
                        es_patience = es_patience_reset
                    if es_patience <= 0:
                        tqdm.write('Early stopping.')
                        break
                    previous_val_loss = avg_loss
            
            if eval_every is not None:
                results = p.starmap(self._evaluate_clients, [(client_split, 'testset') for client_split in client_splits])
                for result in results:
                    for client, loss, tpfptnfn in zip(result[0], result[1], result[2]):
                        results_dict[client][round]['test'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
        
        return results_dict