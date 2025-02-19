import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms
from flib.metrics import average_precision_score
from flib.utils import dataloaders, decrease_lr, filter_args, graphdataset, set_random_seed, tensordatasets
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

class TorchClient():
    """
    PyTorch-specific client for training and evaluation. 
    Can run in isolation and federation.
    """
    def __init__(self, id: str, seed: int, device: str, trainset: str, testset: str, valset_size: float, batch_size: int, Model: Any, optimizer: str, criterion: str, **kwargs):
        self.id = id
        self.seed = seed
        self.device = device
        self.results = {}
        
        set_random_seed(self.seed)
        
        train_df = pd.read_csv(trainset).drop(columns=['account', 'bank'])
        val_df = train_df.sample(frac=valset_size, random_state=seed)
        train_df = train_df.drop(val_df.index)
        test_df = pd.read_csv(testset).drop(columns=['account', 'bank'])
        self.trainset, self.valset, self.testset = tensordatasets(train_df, val_df, test_df, normalize=True, device=self.device)
        y=self.trainset.tensors[1].clone().detach().cpu()
        class_counts = torch.bincount(y)
        class_weights = 1.0 / class_counts
        weights = class_weights[y]
        sampler = WeightedRandomSampler(weights, num_samples=len(y), replacement=True)
        self.trainloader, self.valloader, self.testloader = dataloaders(self.trainset, self.valset, self.testset, batch_size, sampler)
        
        self.model = Model(**filter_args(Model, kwargs)).to(self.device)
        Optimizer = getattr(torch.optim, optimizer)
        self.optimizer = Optimizer(self.model.parameters(), **filter_args(Optimizer, kwargs))
        Criterion = getattr(torch.nn, criterion)
        self.criterion = Criterion(**filter_args(Criterion, kwargs))
    
    def train(self):
        """Train the model on local dataset."""
        self.model.train()
        for x_batch, y_batch in self.trainloader:
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
    
    def evaluate(self, dataset: str = 'trainset') -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model on a given dataset.
        
        Args:
            dataset (str): Dataset name (trainset, valset, testset).
        
        Returns:
            Tuple[float, np.ndarray, np.ndarray]: Loss, predicted logits, ground truth labels.
        """
        dataset_mapping = {
            'trainset': self.trainset,
            'valset': self.valset,
            'testset': self.testset
        }
        dataset = dataset_mapping.get(dataset, self.trainset)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(dataset.tensors[0])
            loss = self.criterion(y_pred, dataset.tensors[1]).item()
        return loss, y_pred.cpu().numpy(), dataset.tensors[1].cpu().numpy()
    
    def run(self, n_rounds: int = 100, eval_every: int = 5, lr_patience: int = 10, es_patience: int = 20, **kwargs) -> Dict:
        """
        Run training and evaluation loop.
        
        Args: 
            n_rounds (int): Number of rounds (aka epochs).
            eval_every (int): Number of rounds between evalualtions.
            lr_patience (int): Learing rate patience.
            es_patience (int): Early stopping patience.
        
        Returns:
            Dict: Results from traning, validation and testing.
        """
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        self.log(dataset='trainset', round=0, loss=loss, y_pred=y_pred, y_true=y_true)
        previous_train_loss = loss
        
        loss, y_pred, y_true = self.evaluate(dataset='valset')
        self.log(dataset='valset', round=0, loss=loss, y_pred=y_pred, y_true=y_true)
        previous_val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
        
        for round in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
            
            set_random_seed(self.seed+round)
            
            self.train()
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            self.log(dataset='trainset', round=round, loss=loss, y_pred=y_pred, y_true=y_true)
            
            if loss >= previous_train_loss:
                lr_patience -= 1
            else:
                lr_patience = lr_patience_reset
            if lr_patience <= 0:
                tqdm.write(f"Decreasing learning rate, round: {round}")
                decrease_lr(self.optimizer, factor=0.5)
                lr_patience = lr_patience_reset
            previous_train_loss = loss
            
            if round % eval_every == 0:
                loss, y_pred, y_true = self.evaluate(dataset='valset')
                self.log(dataset='valset', round=round, loss=loss, y_pred=y_pred, y_true=y_true)
                val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
                if val_average_precision <= previous_val_average_precision:
                    es_patience -= eval_every
                else:
                    es_patience = es_patience_reset
                if es_patience <= 0:
                    tqdm.write(f"Early stopping, round: {round}")
                    break
                previous_val_average_precision = val_average_precision
        
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        self.log(dataset='trainset', y_pred=y_pred, y_true=y_true, metrics=['precision_recall_curve', 'roc_curve'])
        loss, y_pred, y_true = self.evaluate(dataset='valset')
        self.log(dataset='valset', y_pred=y_pred, y_true=y_true, metrics=['precision_recall_curve', 'roc_curve'])
        loss, y_pred, y_true = self.evaluate(dataset='testset')
        self.log(dataset='testset', y_pred=y_pred, y_true=y_true, round=round, loss=loss, metrics=['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'precision_recall_curve', 'roc_curve'])
        
        return self.results
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Retrieve model parameters."""
        return self.model.get_parameters()

    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters."""
        self.model.set_parameters(parameters)

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Retrieve model gradients."""
        return self.model.get_gradients()

    def set_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Set model gradients."""
        self.model.set_gradients(gradients)

    def compute_gradients(self) -> Dict[str, torch.Tensor]:
        """Train and retrive gradients"""
        params_before = self.get_parameters()
        self.train()
        params_after = self.get_parameters()
        gradients = {}
        with torch.no_grad():
            for name in params_before:
                gradients[name] = params_before[name] - params_after[name]
        return gradients
    
    def log(self, dataset: str, y_pred: np.ndarray, y_true: np.ndarray, round: int = None, loss: float = None, metrics: List[str] = None):
        """
        Log training results for a given round.
        
        Args:
            dataset (str): Dataset name.
            y_pred (np.ndarray): Model predictions.
            y_true (np.ndarray): Ground truth labels.
            round (int): Training round.
            loss (float): Loss value.
            metrics (list): List of metrics to calculate.
        """
        if metrics is None:
            metrics = ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall']

        if dataset not in self.results:
            self.results[dataset] = {metric: [] for metric in metrics}
            self.results[dataset]['round'] = []
            self.results[dataset]['loss'] = []

        if round is not None:
            self.results[dataset]['round'].append(round)
        if loss is not None:
            self.results[dataset]['loss'].append(loss)

        for metric in metrics:
            if metric == 'accuracy':
                self.results[dataset]['accuracy'].append(accuracy_score(y_true, (y_pred[:, 1] > 0.5)))
            elif metric == 'average_precision':
                self.results[dataset]['average_precision'].append(average_precision_score(y_true, y_pred[:, 1], recall_span=(0.6, 1.0)))
            elif metric == 'balanced_accuracy':
                self.results[dataset]['balanced_accuracy'].append(balanced_accuracy_score(y_true, (y_pred[:, 1] > 0.5)))
            elif metric == 'f1':
                self.results[dataset]['f1'].append(f1_score(y_true, (y_pred[:, 1] > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'precision':
                self.results[dataset]['precision'].append(precision_score(y_true, (y_pred[:, 1] > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'recall':
                self.results[dataset]['recall'].append(recall_score(y_true, (y_pred[:, 1] > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'precision_recall_curve':
                self.results[dataset]['precision_recall_curve'] = precision_recall_curve(y_true, y_pred[:, 1])
            elif metric == 'roc_curve':
                self.results[dataset]['roc_curve'] = roc_curve(y_true, y_pred[:, 1])

class TorchGeometricClient():
    """
    PyTorchGeometric-specific client for training and evaluation. 
    Can run in isolation and federation.
    """
    def __init__(self, id: str, seed: int, device: str, trainset_nodes: str, testset_nodes: str, trainset_edges: str, testset_edges: str, valset_size: float, Model: Any, optimizer: str, criterion: str, **kwargs):
        self.id = id
        self.seed = seed
        self.device = device
        self.results = {}
        
        set_random_seed(self.seed)
        
        train_nodes_df = pd.read_csv(trainset_nodes).drop(columns=['bank']).rename(columns={'account': 'node'})
        train_edges_df = pd.read_csv(trainset_edges)
        test_nodes_df = pd.read_csv(testset_nodes).drop(columns=['bank']).rename(columns={'account': 'node'})
        test_edges_df = pd.read_csv(testset_edges)
        
        self.trainset, self.testset = graphdataset(train_nodes_df, train_edges_df, test_nodes_df, test_edges_df, device=device)
        self.trainset = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=valset_size, num_test=0)(self.trainset)
        
        self.model = Model(**filter_args(Model, kwargs)).to(self.device)
        Optimizer = getattr(torch.optim, optimizer)
        self.optimizer = Optimizer(self.model.parameters(), **filter_args(Optimizer, kwargs))
        Criterion = getattr(torch.nn, criterion)
        class_counts = torch.bincount(self.trainset.y)
        weight = class_counts.max() / class_counts
        self.criterion = Criterion(weight = weight, **filter_args(Criterion, kwargs))
    
    def train(self):
        """Train the model on local dataset."""
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(self.trainset)
        loss = self.criterion(y_pred[self.trainset.train_mask], self.trainset.y[self.trainset.train_mask])
        loss.backward()
        self.optimizer.step()
    
    def evaluate(self, dataset: str = 'trainset') -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate model on a given dataset.
        
        Args:
            dataset (str): Dataset name (trainset, valset, testset).
        
        Returns:
            Tuple[float, np.ndarray, np.ndarray]: Loss, predicted logits, ground truth labels.
        """
        if dataset == 'trainset':
            dataset = self.trainset
            mask = dataset.train_mask
        elif dataset == 'valset':
            dataset = self.trainset
            mask = dataset.val_mask
        elif dataset == 'testset':
            dataset = self.testset
            mask = torch.tensor([True] * len(dataset.y))
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(dataset)
            loss = self.criterion(y_pred[mask], dataset.y[mask]).item()
        return loss, y_pred.cpu().numpy(), dataset.y.cpu().numpy()
    
    def run(self, n_rounds: int = 100, eval_every: int = 5, lr_patience: int = 10, es_patience: int = 20, **kwargs) -> Dict:
        """
        Run training and evaluation loop.
        
        Args: 
            n_rounds (int): Number of rounds (aka epochs).
            eval_every (int): Number of rounds between evalualtions.
            lr_patience (int): Learing rate patience.
            es_patience (int): Early stopping patience.
        
        Returns:
            Dict: Results from traning, validation and testing.
        """
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        self.log(dataset='trainset', round=0, loss=loss, y_pred=y_pred, y_true=y_true)
        previous_train_loss = loss
        
        loss, y_pred, y_true = self.evaluate(dataset='valset')
        self.log(dataset='valset', round=0, loss=loss, y_pred=y_pred, y_true=y_true)
        previous_val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
        
        for round in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
            
            set_random_seed(self.seed+round)
            
            self.train()
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            self.log(dataset='trainset', round=round, loss=loss, y_pred=y_pred, y_true=y_true)
            
            if loss >= previous_train_loss:
                lr_patience -= 1
            else:
                lr_patience = lr_patience_reset
            if lr_patience <= 0:
                tqdm.write(f"Decreasing learning rate, round: {round}")
                decrease_lr(self.optimizer, factor=0.5)
                lr_patience = lr_patience_reset
            previous_train_loss = loss
            
            if round % eval_every == 0:
                loss, y_pred, y_true = self.evaluate(dataset='valset')
                self.log(dataset='valset', round=round, loss=loss, y_pred=y_pred, y_true=y_true)
                val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
                if val_average_precision <= previous_val_average_precision:
                    es_patience -= eval_every
                else:
                    es_patience = es_patience_reset
                if es_patience <= 0:
                    tqdm.write(f"Early stopping, round: {round}")
                    break
                previous_val_average_precision = val_average_precision
        
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        self.log(dataset='trainset', y_pred=y_pred, y_true=y_true, metrics=['precision_recall_curve', 'roc_curve'])
        loss, y_pred, y_true = self.evaluate(dataset='valset')
        self.log(dataset='valset', y_pred=y_pred, y_true=y_true, metrics=['precision_recall_curve', 'roc_curve'])
        loss, y_pred, y_true = self.evaluate(dataset='testset')
        self.log(dataset='testset', y_pred=y_pred, y_true=y_true, round=round, loss=loss, metrics=['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall', 'precision_recall_curve', 'roc_curve'])
        
        return self.results
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Retrieve model parameters."""
        return self.model.get_parameters()

    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters."""
        self.model.set_parameters(parameters)

    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Retrieve model gradients."""
        return self.model.get_gradients()

    def set_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Set model gradients."""
        self.model.set_gradients(gradients)

    def compute_gradients(self) -> Dict[str, torch.Tensor]:
        """Train and retrive gradients"""
        params_before = self.get_parameters()
        self.train()
        params_after = self.get_parameters()
        gradients = {}
        with torch.no_grad():
            for name in params_before:
                gradients[name] = params_before[name] - params_after[name]
        return gradients
    
    def log(self, dataset: str, y_pred: np.ndarray, y_true: np.ndarray, round: int = None, loss: float = None, metrics: List[str] = None):
        """
        Log training results for a given round.
        
        Args:
            dataset (str): Dataset name.
            y_pred (np.ndarray): Model predictions.
            y_true (np.ndarray): Ground truth labels.
            round (int): Training round.
            loss (float): Loss value.
            metrics (list): List of metrics to calculate.
        """
        if metrics is None:
            metrics = ['accuracy', 'average_precision', 'balanced_accuracy', 'f1', 'precision', 'recall']

        if dataset not in self.results:
            self.results[dataset] = {metric: [] for metric in metrics}
            self.results[dataset]['round'] = []
            self.results[dataset]['loss'] = []

        if round is not None:
            self.results[dataset]['round'].append(round)
        if loss is not None:
            self.results[dataset]['loss'].append(loss)

        for metric in metrics:
            if metric == 'accuracy':
                self.results[dataset]['accuracy'].append(accuracy_score(y_true, (y_pred[:, 1] > 0.5)))
            elif metric == 'average_precision':
                self.results[dataset]['average_precision'].append(average_precision_score(y_true, y_pred[:, 1], recall_span=(0.6, 1.0)))
            elif metric == 'balanced_accuracy':
                self.results[dataset]['balanced_accuracy'].append(balanced_accuracy_score(y_true, (y_pred[:, 1] > 0.5)))
            elif metric == 'f1':
                self.results[dataset]['f1'].append(f1_score(y_true, (y_pred[:, 1] > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'precision':
                self.results[dataset]['precision'].append(precision_score(y_true, (y_pred[:, 1] > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'recall':
                self.results[dataset]['recall'].append(recall_score(y_true, (y_pred[:, 1] > 0.5), pos_label=1, zero_division=0.0))
            elif metric == 'precision_recall_curve':
                self.results[dataset]['precision_recall_curve'] = precision_recall_curve(y_true, y_pred[:, 1])
            elif metric == 'roc_curve':
                self.results[dataset]['roc_curve'] = roc_curve(y_true, y_pred[:, 1])