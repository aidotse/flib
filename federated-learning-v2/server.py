import multiprocessing as mp
import time
import torch
from collections import OrderedDict
import copy
import numpy as np

import time

class Server():
    def __init__(self, clients, model, n_workers, log_file='log', log_predictions=False):
        self.model = model
        self.clients = clients
        self.n_workers = n_workers
        self.log_predictions = log_predictions
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            f.write('type;round;client;loss;y_pred;y_true\n')

    def _train_clients(self, clients):
        models = []
        client_names = []
        losses = []
        y_preds = []
        y_trues = []
        for client in clients:
            train_loss, y_pred, y_true = client.train(return_predictions=self.log_predictions)
            models.append(client.model())
            client_names.append(client.name)
            losses.append(train_loss)
            y_preds.append(y_pred)
            y_trues.append(y_true)
        return client_names, losses, y_preds, y_trues, models
    
    def _validate_clients(self, clients):
        for client in clients:
            val_loss, y_pred, y_true = client.validate(return_predictions=self.log_predictions)
            #logging.info(f'{client.name}: val_loss={val_loss}, y_pred={y_pred}, y_true={y_true}')

    def _test_clients(self, clients):
        client_names = []
        losses = []
        y_preds = []
        y_trues = []
        for client in clients:
            test_loss, y_pred, y_true = client.test(return_predictions=self.log_predictions)
            client_names.append(client.name)
            losses.append(test_loss)
            y_preds.append(y_pred)
            y_trues.append(y_true)
            #logging.info(f'{client.name}: test_loss={test_loss}, y_pred={y_pred}, y_true={y_true}')
        return client_names, losses, y_preds, y_trues
            
            
    def _average_models(self, models, weights=None):
        if weights:
            weights = [weight/sum(weights) for weight in weights]
        else:
            weights = [1.0/len(models) for _ in models]
        avg_models = OrderedDict([(key, 0.0) for key in models[0].keys()])
        for key in models[0].keys():
            for model, weight in zip(models, weights):
                avg_models[key] += torch.mul(model[key], weight) 
        return avg_models

    def run(self, n_rounds=30, eval_every=10, n_rounds_no_aggregation=0):

        with mp.Pool(self.n_workers) as p:
                
            for client in self.clients:
                client.load_model(copy.deepcopy(self.model))
            
            client_splits = np.array_split(self.clients, self.n_workers)
            
            avg_losses = []

            for round in range(n_rounds):
                
                round_time = time.time()
                
                results = p.map(self._train_clients, client_splits)
                models = [model for sublist in results for model in sublist[4]]
                self.log('train', round, results)

                if round >= n_rounds_no_aggregation:
                    self.model = self._average_models(models)
                    for client in self.clients:
                        client.load_model(copy.deepcopy(self.model))
                
                if round % eval_every == 0:
                    results = p.map(self._test_clients, client_splits)
                    self.log('test', round, results)
                    losses = [loss for sublist in results for loss in sublist[1]]
                    avg_loss = sum(losses)/len(losses)
                    avg_losses.append(avg_loss)

                round_time = time.time() - round_time
                
                print(' progress: [%s%s], round: %i/%i, time left: ~%.2f min   ' % ('#' * (round * 80 // (n_rounds-1)), '.' * (80 - round * 80 // (n_rounds-1)), round, n_rounds-1, (n_rounds - 1 - round) * round_time / 60), end='\r')
        
        return avg_losses            

    def log(self, type, round, results):
        client_names  = [name for sublist in results for name in sublist[0]]
        losses = [loss for sublist in results for loss in sublist[1]]
        y_preds = [y_pred for sublist in results for y_pred in sublist[2]]
        y_trues = [y_true for sublist in results for y_true in sublist[3]]
        with open(self.log_file, 'a') as f:
            for client_name, loss, y_pred, y_true in zip(client_names, losses, y_preds, y_trues):
                f.write(f'{type};{round};{client_name};{loss};{y_pred};{y_true}\n')

