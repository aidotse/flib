import torch 
from sklearn.metrics import confusion_matrix, roc_curve
from flib.utils import tensordatasets, dataloaders, decrease_lr
from flib.train.models import LogisticRegressor
from flib.train import criterions
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class LogRegClient():
    def __init__(self, name:str, train_df:pd.DataFrame, val_df:pd.DataFrame=None, test_df:pd.DataFrame=None, device:str='cpu', batch_size=64, optimizer='SGD', criterion='ClassBalancedLoss', lr=0.01, **kwargs):
        self.name = name
        self.device = device
        
        self.trainset, self.valset, self.testset = tensordatasets(train_df, val_df, test_df, normalize=True, device=self.device)
        self.trainloader, self.valloader, self.testloader = dataloaders(self.trainset, self.valset, self.testset, batch_size)
        
        input_dim = self.trainset.tensors[0].shape[1]
        output_dim = self.trainset.tensors[1].unique().shape[0]
        self.model = LogisticRegressor(input_dim=input_dim, output_dim=output_dim).to(self.device)
        
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=lr)
        n_samples_per_classes = [sum(self.trainset.tensors[1] == 0).detach().cpu().numpy(), sum(self.trainset.tensors[1] == 1).detach().cpu().numpy()]
        self.criterion = getattr(criterions, criterion)(n_samples_per_classes=n_samples_per_classes, **kwargs)

    def train(self, state_dict=None):
        if state_dict:
            self.model.load_state_dict(state_dict)
        self.model.train()
        loss = 0.0
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        for x_batch, y_batch in self.trainloader:
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            l = self.criterion(y_pred, y_batch)
            l.backward()
            self.optimizer.step()
            loss += l.item() / len(self.trainloader)
            for threshold in range(0, 101):
                cm = confusion_matrix(y_batch.cpu(), (y_pred[:,1] > (threshold / 100)).to(torch.int64).cpu(), labels=[0, 1], normalize='all')
                tpfptnfn[threshold]['tp'] += cm[1,1] / len(self.trainloader)
                tpfptnfn[threshold]['fp'] += cm[0,1] / len(self.trainloader)
                tpfptnfn[threshold]['tn'] += cm[0,0] / len(self.trainloader)
                tpfptnfn[threshold]['fn'] += cm[1,0] / len(self.trainloader)
        return loss, tpfptnfn

    def evaluate(self, state_dict=None, dataset='testset'):
        if state_dict:
            self.model.load_state_dict(state_dict)
        if dataset == 'trainset':
            dataloader = self.trainloader
        elif dataset == 'valset':
            dataloader = self.valloader
        elif dataset == 'testset':
            dataloader = self.testloader
        self.model.eval()
        loss = 0.0
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                y_pred = self.model(x_batch)
                loss += self.criterion(y_pred, y_batch).item() / len(dataloader)
                for threshold in range(0, 101):
                    cm = confusion_matrix(y_batch.cpu(), (y_pred[:,1] > (threshold / 100)).to(torch.int64).cpu(), labels=[0, 1], normalize='all')
                    tpfptnfn[threshold]['tp'] += cm[1,1] / len(dataloader)
                    tpfptnfn[threshold]['fp'] += cm[0,1] / len(dataloader)
                    tpfptnfn[threshold]['tn'] += cm[0,0] / len(dataloader)
                    tpfptnfn[threshold]['fn'] += cm[1,0] / len(dataloader)
        return loss, tpfptnfn
    
    def run(self, state_dict=None, n_epochs=100, eval_every=10, lr_patience=5, es_patience=15, **kwargs):
        if state_dict:
            self.model.load_state_dict(state_dict)
        
        results_dict = {0: {}}
        loss, tpfptnfn = self.evaluate(dataset='trainset')
        results_dict[0]['train'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
        previous_loss = loss
        if eval_every is not None:
            if self.valset is not None:
                loss, tpfptnfn = self.evaluate(dataset='valset')
                results_dict[0]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}

        for epoch in tqdm(range(1, n_epochs+1), desc='progress', leave=False):
            
            loss, tpfptnfn = self.train()
            results_dict[epoch] = {'train': {'loss': loss, 'tpfptnfn': tpfptnfn}}
            
            if loss >= previous_loss - 0.0005:
                lr_patience -= 1
                es_patience -= 1
            else:
                lr_patience = 5
                es_patience = 15
            previous_loss = loss
            
            if eval_every is not None and epoch % eval_every == 0:
                if self.valset is not None:
                    loss, tpfptnfn = self.evaluate(dataset='valset')
                    results_dict[epoch]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
            
            if lr_patience <= 0:
                decrease_lr(self.optimizer, factor=0.5)
            
            if es_patience <= 0 and (eval_every is None or epoch % eval_every == 0):
                break
        
        if eval_every is not None and self.testset is not None:
            loss, tpfptnfn = self.evaluate(dataset='testset')
            results_dict[epoch]['test'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
                
        return results_dict

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            state_dict[key] = value.to(self.device)
        self.model.load_state_dict(state_dict)
    
    def get_state_dict(self):
        model = self.model.state_dict()
        for key, value in model.items():
            model[key] = value.detach().cpu()
        return model

class OldLogRegClient():
    def __init__(self, name:str, train_df:pd.DataFrame, val_df:pd.DataFrame=None, test_df:pd.DataFrame=None, device:str='cpu', n_epochs:int=30, eval_every:int=10, batch_size:int=64, optimizer:str='SGD', criterion:str='ClassBalancedLoss', lr:float=0.01, **kwargs):
        self.name = name
        self.device = device
        self.n_epochs = n_epochs
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.lr = lr
        
        self.trainset, self.valset, self.testset = tensordatasets(train_df, val_df, test_df, normalize=True, device=self.device)
        self.trainloader, self.valloader, self.testloader = dataloaders(self.trainset, self.valset, self.testset, self.batch_size)
        input_dim = self.trainset.tensors[0].shape[1]
        output_dim = self.trainset.tensors[1].unique().shape[0]
        
        self.model = LogisticRegressor(input_dim=input_dim, output_dim=output_dim).to(self.device)
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=self.lr)
        if criterion == 'ClassBalancedLoss':
            n_samples_per_classes = [sum(self.trainset.tensors[1] == 0).detach().cpu().numpy(), sum(self.trainset.tensors[1] == 1).detach().cpu().numpy()]
            self.criterion = getattr(criterions, criterion)(beta=kwargs.get('beta', 0.9), n_samples_per_classes=n_samples_per_classes, loss_type='sigmoid')
        else:
            self.criterion = getattr(criterions, criterion)()

    def run(self, state_dict=None):
        lr_patience = 5
        es_patience = 15
        
        if state_dict:
            self.model.load_state_dict(state_dict)
        
        results_dict = {0: {}}
        loss, tpfptnfn = self.evaluate(dataset='trainset')
        results_dict[0]['train'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
        previous_loss = loss
        if self.eval_every is not None:
            if self.valset is not None:
                loss, tpfptnfn = self.evaluate(dataset='valset')
                results_dict[0]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}

        for epoch in range(1, self.n_epochs+1):
            
            loss, tpfptnfn = self.train()
            results_dict[epoch] = {'train': {'loss': loss, 'tpfptnfn': tpfptnfn}}
            
            if loss >= previous_loss - 0.0005:
                lr_patience -= 1
                es_patience -= 1
            else:
                lr_patience = 5
                es_patience = 15
            previous_loss = loss
            
            if self.eval_every is not None and epoch % self.eval_every == 0:
                if self.valset is not None:
                    loss, tpfptnfn = self.evaluate(dataset='valset')
                    results_dict[epoch]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
            
            
            if lr_patience <= 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                self.lr *= 0.5
            
            if es_patience <= 0 and (self.eval_every is None or epoch % self.eval_every == 0):
                break
        
        if self.eval_every is not None and self.testset is not None:
            loss, tpfptnfn = self.evaluate(dataset='testset')
            results_dict[epoch]['test'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
                
        return results_dict

    def train(self, state_dict=None):
        if state_dict:
            self.model.load_state_dict(state_dict)
        self.model.train()
        loss = 0.0
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        for x_batch, y_batch in self.trainloader:
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            l = self.criterion(y_pred, y_batch)
            l.backward()
            self.optimizer.step()
            loss += l.item() / len(self.trainloader)
            for threshold in range(0, 101):
                cm = confusion_matrix(y_batch.cpu(), (y_pred[:,1] > (threshold / 100)).to(torch.int64).cpu(), labels=[0, 1], normalize='all')
                tpfptnfn[threshold]['tp'] += cm[1,1] / len(self.trainloader)
                tpfptnfn[threshold]['fp'] += cm[0,1] / len(self.trainloader)
                tpfptnfn[threshold]['tn'] += cm[0,0] / len(self.trainloader)
                tpfptnfn[threshold]['fn'] += cm[1,0] / len(self.trainloader)
        return loss, tpfptnfn
    
    def evaluate(self, state_dict=None, dataset='testset'):
        if state_dict:
            self.model.load_state_dict(state_dict)
        if dataset == 'trainset':
            dataloader = self.trainloader
        elif dataset == 'valset':
            dataloader = self.valloader
        elif dataset == 'testset':
            dataloader = self.testloader
        self.model.eval()
        loss = 0.0
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                y_pred = self.model(x_batch)
                loss += self.criterion(y_pred, y_batch).item() / len(dataloader)
                for threshold in range(0, 101):
                    cm = confusion_matrix(y_batch.cpu(), (y_pred[:,1] > (threshold / 100)).to(torch.int64).cpu(), labels=[0, 1], normalize='all')
                    tpfptnfn[threshold]['tp'] += cm[1,1] / len(dataloader)
                    tpfptnfn[threshold]['fp'] += cm[0,1] / len(dataloader)
                    tpfptnfn[threshold]['tn'] += cm[0,0] / len(dataloader)
                    tpfptnfn[threshold]['fn'] += cm[1,0] / len(dataloader)
        return loss, tpfptnfn
    
    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            state_dict[key] = value.to(self.device)
        self.model.load_state_dict(state_dict)
    
    def get_state_dict(self):
        model = self.model.state_dict()
        for key, value in model.items():
            model[key] = value.detach().cpu()
        return model

class DecisionTreeClient():
    def __init__(self, name:str, train_df:pd.DataFrame, val_df:pd.DataFrame=None, test_df:pd.DataFrame=None, **kwargs):
        self.name = name
        valset_size = kwargs.get('valset_size', None)
        if val_df is None and valset_size is not None:
            val_df = train_df.sample(frac=valset_size, random_state=42)
            train_df = train_df.drop(val_df.index)
        self.X_train = train_df.drop(columns=['account', 'bank', 'is_sar']).to_numpy()
        self.y_train = train_df['is_sar'].to_numpy()
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        if val_df is not None:
            self.X_val = val_df.drop(columns=['account', 'bank', 'is_sar']).to_numpy()
            self.X_val = scaler.transform(self.X_val)
            self.y_val = val_df['is_sar'].to_numpy()
        else:
            self.X_val = None
            self.y_val = None
        if test_df is not None:
            self.X_test = test_df.drop(columns=['account', 'bank', 'is_sar']).to_numpy()
            self.X_test = scaler.transform(self.X_test)
            self.y_test = test_df['is_sar'].to_numpy()
        else:
            self.X_test = None
            self.y_test = None
        
        self.model = DecisionTreeClassifier(
            criterion=kwargs.get('criterion', 'gini'),
            splitter=kwargs.get('splitter', 'best'),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            min_weight_fraction_leaf=kwargs.get('min_weight_fraction_leaf', 0.0),
            max_features=kwargs.get('max_features', None),
            max_leaf_nodes=kwargs.get('max_leaf_nodes', None),
            min_impurity_decrease=kwargs.get('min_impurity_decrease', 0.0),
            class_weight=kwargs.get('class_weight', 'balanced'),
            random_state=kwargs.get('random_state', 42)
        )
    
    def run(self):
        self.train()
        _, train_tpfptnfn = self.evaluate(dataset='trainset')
        _, val_tpfptnfn = self.evaluate(dataset='valset')
        _, test_tpfptnfn = self.evaluate(dataset='testset')
        results = {0: {'train': {'loss': None, 'tpfptnfn': train_tpfptnfn}, 'val': {'loss': None, 'tpfptnfn': val_tpfptnfn}, 'test': {'loss': None, 'tpfptnfn': test_tpfptnfn}}}
        return results
    
    def train(self, tune_hyperparameters=False):
        if tune_hyperparameters:
            pass
        else:    
            self.model.fit(self.X_train, self.y_train)
        return
    
    def evaluate(self, dataset='trainset'):
        if dataset == 'trainset':
            X, y = self.X_train, self.y_train
        elif dataset == 'valset':
            if self.X_val is None:
                return None, None
            else:
                X, y = self.X_val, self.y_val
        elif dataset == 'testset':
            if self.X_test is None:
                return None, None
            else:
                X, y = self.X_test, self.y_test
        y_pred = self.model.predict_proba(X)
        curve = roc_curve(y, y_pred[:,1])
        print(curve)
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        for threshold in range(0, 101):
            cm = confusion_matrix(y, (y_pred[:,1] > (threshold / 100)), labels=[0, 1], normalize='all')
            tpfptnfn[threshold]['tp'] = cm[1,1]
            tpfptnfn[threshold]['fp'] = cm[0,1]
            tpfptnfn[threshold]['tn'] = cm[0,0]
            tpfptnfn[threshold]['fn'] = cm[1,0]
        return None, tpfptnfn
    
    def get_state_dict(self):
        return None
    
    def load_state_dict(self, state_dict):
        return