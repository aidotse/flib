import multiprocessing as mp
import time
import logging
import torch
from collections import OrderedDict
import copy
import numpy as np

class Server():
    def __init__(self, clients, model, n_workers):
        self.model = model
        self.clients = clients
        self.n_workers = n_workers

    def _train_clients(self, clients):
        models = []
        for client in clients:
            loss = client.train(return_metrics=False)
            #logging.info(f'{client.name}: train_loss={loss}')
            #logging.info(f'{client.name}: train_loss={loss}, train_accuracy={accuracy}, train_precision={precision}, train_recall={recall}, train_f1={f1}, train_cf_matrix={cf_matrix}')
            model = client.model()
            models.append(model)
        return models
    
    def _validate_clients(self, clients):
        for client in clients:
            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_cf_matrix = client.validate(return_metrics=True)
            logging.info(f'{client.name}: val_loss={val_loss}, val_accuracy={val_accuracy}, val_precision={val_precision}, val_recall={val_recall}, val_f1={val_f1}, val_cf_matrix={val_cf_matrix}')

    def _test_clients(self, clients):
        for client in clients:
            test_loss, test_accuracy, test_precision, test_recall, test_f1, test_cf_matrix = client.test(return_metrics=True)
            logging.info(f'{client.name}: test_loss={test_loss}, test_accuracy={test_accuracy}, test_precision={test_precision}, test_recall={test_recall}, test_f1={test_f1}, test_cf_matrix={test_cf_matrix}')

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

    def run(self, n_rounds=30, eval_every=10):

        mp.set_start_method('spawn')

        with mp.Pool(self.n_workers) as p:
                
            for client in self.clients:
                client.load_model(copy.deepcopy(self.model))
            
            client_splits = np.array_split(self.clients, self.n_workers)

            for round in range(n_rounds):
                
                round_time = time.time()
                
                logging.info(f'round {round}')
                
                models = p.map(self._train_clients, client_splits)
                models = [model for sublist in models for model in sublist]

                self.model = self._average_models(models)
                
                for client in self.clients:
                    client.load_model(copy.deepcopy(self.model))
                
                if round % eval_every == 0:
                    p.map(self._test_clients, client_splits)

                round_time = time.time() - round_time
                
                print(' progress: [%s%s], round: %i/%i, time left: ~%.2f min   ' % ('#' * (round * 80 // (n_rounds-1)), '.' * (80 - round * 80 // (n_rounds-1)), round, n_rounds-1, (n_rounds - 1 - round) * round_time / 60), end='\r')
            


