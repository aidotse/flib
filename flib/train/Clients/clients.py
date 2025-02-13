import torch 
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, PrecisionRecallDisplay, RocCurveDisplay
from flib import utils
from flib.utils import tensordatasets, dataloaders, decrease_lr
from flib.train.models import LogisticRegressor, MLP, GraphSAGE
from flib.train import criterions
from flib.train.metrics import calculate_average_precision
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
import torch_geometric
import yaml


class LogRegClient():
    def __init__(self, name:str, seed:int, device:str, nodes_train:str, nodes_test:str, valset_size:float, batch_size:int, optimizer:str, lr: float, weight_decay: float, criterion:str, gamma:float, **kwargs):
        self.name = name
        self.device = device

        train_df = pd.read_csv(nodes_train).drop(columns=['account', 'bank'])
        val_df = train_df.sample(frac=valset_size, random_state=seed)
        train_df = train_df.drop(val_df.index)
        test_df = pd.read_csv(nodes_test).drop(columns=['account', 'bank'])
        
        self.trainset, self.valset, self.testset = tensordatasets(train_df, val_df, test_df, normalize=True, device=self.device)
        
        # Calculate class weights
        y=self.trainset.tensors[1].clone().detach().cpu()
        class_counts = torch.bincount(y)  # y is the list of labels
        class_weights = 1.0 / class_counts
        weights = class_weights[y]
        sampler = WeightedRandomSampler(weights, num_samples=len(y), replacement=True)
        
        self.trainloader, self.valloader, self.testloader = dataloaders(self.trainset, self.valset, self.testset, batch_size, sampler)

        input_dim = self.trainset.tensors[0].shape[1]
        output_dim = self.trainset.tensors[1].unique().shape[0]
        self.model = LogisticRegressor(input_dim=input_dim, output_dim=output_dim).to(self.device)
        
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if criterion == 'ClassBalancedLoss':
            n_samples_per_classes = [sum(self.trainset.tensors[1] == 0).detach().cpu().numpy(), sum(self.trainset.tensors[1] == 1).detach().cpu().numpy()]
            self.criterion = criterions.ClassBalancedLoss(n_samples_per_classes=n_samples_per_classes, gamma=gamma)
        else:
            self.criterion = getattr(torch.nn, criterion)()

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
                cm = confusion_matrix(y_batch.cpu(), (y_pred[:,1] >= (threshold / 100)).to(torch.int64).cpu(), labels=[0, 1])
                tpfptnfn[threshold]['tp'] += cm[1,1]
                tpfptnfn[threshold]['fp'] += cm[0,1]
                tpfptnfn[threshold]['tn'] += cm[0,0]
                tpfptnfn[threshold]['fn'] += cm[1,0]
        return loss, tpfptnfn

    def evaluate(self, state_dict=None, dataset='testset'):
        if state_dict:
            self.model.load_state_dict(state_dict)
        if dataset == 'trainset':
            dataset = self.trainset
        elif dataset == 'valset':
            dataset = self.valset
        elif dataset == 'testset':
            if self.testloader == None:
                return None, None
            dataset = self.testset
        self.model.eval()
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        with torch.no_grad():
            y_pred = self.model(dataset.tensors[0])
            loss = self.criterion(y_pred, dataset.tensors[1]).item()
            for threshold in range(0, 101):
                cm = confusion_matrix(dataset.tensors[1].cpu(), (y_pred[:,1] >= (threshold / 100)).to(torch.int64).cpu(), labels=[0, 1])
                tpfptnfn[threshold]['tp'] = cm[1,1]
                tpfptnfn[threshold]['fp'] = cm[0,1]
                tpfptnfn[threshold]['tn'] = cm[0,0]
                tpfptnfn[threshold]['fn'] = cm[1,0]
        return loss, tpfptnfn
    
    def run(self, state_dict=None, n_rounds=100, eval_every=10, lr_patience=5, es_patience=20, **kwargs):
        if state_dict:
            self.model.load_state_dict(state_dict)
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        
        results_dict = {0: {}}
        loss, tpfptnfn = self.evaluate(dataset='trainset')
        results_dict[0]['train'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
        previous_train_loss = loss
        if eval_every is not None and self.valset is not None:
            loss, tpfptnfn = self.evaluate(dataset='valset')
            results_dict[0]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
            previous_val_average_precision = calculate_average_precision(tpfptnfn, (0.6, 1.0))

        for epoch in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
            
            loss, tpfptnfn = self.train()
            results_dict[epoch] = {'train': {'loss': loss, 'tpfptnfn': tpfptnfn}}
            if loss >= previous_train_loss:
                lr_patience -= 1
            else:
                lr_patience = lr_patience_reset
            if lr_patience <= 0:
                tqdm.write('Decreasing learning rate.')
                decrease_lr(self.optimizer, factor=0.5)
                lr_patience = lr_patience_reset
            previous_train_loss = loss
            
            if eval_every is not None and epoch % eval_every == 0 and self.valset is not None:
                loss, tpfptnfn = self.evaluate(dataset='valset')
                results_dict[epoch]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
                val_average_precision = calculate_average_precision(tpfptnfn, (0.6, 1.0))
                if val_average_precision <= previous_val_average_precision:
                    es_patience -= eval_every
                else:
                    es_patience = es_patience_reset
                if es_patience <= 0:
                    tqdm.write('Early stopping.')
                    break
                previous_val_average_precision = val_average_precision
        
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


class DecisionTreeClient():
    def __init__(self, name:str, seed:int, nodes_train:str, nodes_test:str, valset_size:float, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0, class_weight='balanced', random_state=42, **kwargs):
        self.name = name
        
        train_df = pd.read_csv(nodes_train).drop(columns=['account', 'bank'])
        val_df = train_df.sample(frac=valset_size, random_state=seed)
        train_df = train_df.drop(val_df.index)
        test_df = pd.read_csv(nodes_test).drop(columns=['account', 'bank'])
        
        self.X_train = train_df.drop(columns=['is_sar']).to_numpy()
        self.y_train = train_df['is_sar'].to_numpy()
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = val_df.drop(columns=['is_sar']).to_numpy()
        self.X_val = scaler.transform(self.X_val)
        self.y_val = val_df['is_sar'].to_numpy()
        self.X_test = test_df.drop(columns=['is_sar']).to_numpy()
        self.X_test = scaler.transform(self.X_test)
        self.y_test = test_df['is_sar'].to_numpy()
        
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            random_state=random_state
        )
    
    def run(self, **kwargs):
        self.train()
        train_loss, train_tpfptnfn = self.evaluate(dataset='trainset')
        val_loss, val_tpfptnfn = self.evaluate(dataset='valset')
        test_loss, test_tpfptnfn = self.evaluate(dataset='testset')
        results = {0: {'train': {'loss': train_loss, 'tpfptnfn': train_tpfptnfn}, 'val': {'loss': val_loss, 'tpfptnfn': val_tpfptnfn}, 'test': {'loss': test_loss, 'tpfptnfn': test_tpfptnfn}}}
        return results
    
    def train(self):
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
        #roc_auc = roc_auc_score(y, y_pred[:,1])
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        for threshold in range(0, 101):
            cm = confusion_matrix(y, (y_pred[:,1] >= (threshold / 100)), labels=[0, 1])
            tpfptnfn[threshold]['tp'] = cm[1,1]
            tpfptnfn[threshold]['fp'] = cm[0,1]
            tpfptnfn[threshold]['tn'] = cm[0,0]
            tpfptnfn[threshold]['fn'] = cm[1,0]
        return None, tpfptnfn #-roc_auc, tpfptnfn
    
    def get_state_dict(self):
        return None
    
    def load_state_dict(self, state_dict):
        return


class RandomForestClient():
    def __init__(self, name:str, seed:str, nodes_train:str, nodes_test:str, valset_size:float, n_estimators=100, criterion='gini', max_depth=None, class_weight='balanced', random_state=42, **kwargs):
        self.name = name
        
        train_df = pd.read_csv(nodes_train).drop(columns=['account', 'bank'])
        val_df = train_df.sample(frac=valset_size, random_state=seed)
        train_df = train_df.drop(val_df.index)
        test_df = pd.read_csv(nodes_test).drop(columns=['account', 'bank'])
        
        self.X_train = train_df.drop(columns=['is_sar']).to_numpy()
        self.y_train = train_df['is_sar'].to_numpy()
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = val_df.drop(columns=['is_sar']).to_numpy()
        self.X_val = scaler.transform(self.X_val)
        self.y_val = val_df['is_sar'].to_numpy()
        self.X_test = test_df.drop(columns=['is_sar']).to_numpy()
        self.X_test = scaler.transform(self.X_test)
        self.y_test = test_df['is_sar'].to_numpy()
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state
        )
    
    def run(self, **kwargs):
        self.train()
        train_loss, train_tpfptnfn = self.evaluate(dataset='trainset')
        val_loss, val_tpfptnfn = self.evaluate(dataset='valset')
        test_loss, test_tpfptnfn = self.evaluate(dataset='testset')
        results = {0: {'train': {'loss': train_loss, 'tpfptnfn': train_tpfptnfn}, 'val': {'loss': val_loss, 'tpfptnfn': val_tpfptnfn}, 'test': {'loss': test_loss, 'tpfptnfn': test_tpfptnfn}}}
        return results
    
    def train(self):
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
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        for threshold in range(0, 101):
            cm = confusion_matrix(y, (y_pred[:,1] >= (threshold / 100)), labels=[0, 1])
            tpfptnfn[threshold]['tp'] = cm[1,1]
            tpfptnfn[threshold]['fp'] = cm[0,1]
            tpfptnfn[threshold]['tn'] = cm[0,0]
            tpfptnfn[threshold]['fn'] = cm[1,0]
        return None, tpfptnfn
    
    def get_state_dict(self):
        return None
    
    def load_state_dict(self, state_dict):
        return
    

class GradientBoostingClient():
    def __init__(self, name:str, seed:int, nodes_train:str, nodes_test:str, valset_size:float, loss='log_loss', learning_rate=0.1, n_estimators=100, criterion='friedman_mse', max_depth=3, random_state=42, **kwargs):
        self.name = name
        
        train_df = pd.read_csv(nodes_train).drop(columns=['account', 'bank'])
        val_df = train_df.sample(frac=valset_size, random_state=seed)
        train_df = train_df.drop(val_df.index)
        test_df = pd.read_csv(nodes_test).drop(columns=['account', 'bank'])
        
        self.X_train = train_df.drop(columns=['is_sar']).to_numpy()
        self.y_train = train_df['is_sar'].to_numpy()
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = val_df.drop(columns=['is_sar']).to_numpy()
        self.X_val = scaler.transform(self.X_val)
        self.y_val = val_df['is_sar'].to_numpy()
        self.X_test = test_df.drop(columns=['is_sar']).to_numpy()
        self.X_test = scaler.transform(self.X_test)
        self.y_test = test_df['is_sar'].to_numpy()
        
        self.model = GradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            random_state=random_state
        )
    
    def run(self, **kwargs):
        self.train()
        train_loss, train_tpfptnfn = self.evaluate(dataset='trainset')
        val_loss, val_tpfptnfn = self.evaluate(dataset='valset')
        test_loss, test_tpfptnfn = self.evaluate(dataset='testset')
        results = {0: {'train': {'loss': train_loss, 'tpfptnfn': train_tpfptnfn}, 'val': {'loss': val_loss, 'tpfptnfn': val_tpfptnfn}, 'test': {'loss': test_loss, 'tpfptnfn': test_tpfptnfn}}}
        return results
    
    def train(self):
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
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        for threshold in range(0, 101):
            cm = confusion_matrix(y, (y_pred[:,1] >= (threshold / 100)), labels=[0, 1])
            tpfptnfn[threshold]['tp'] = cm[1,1]
            tpfptnfn[threshold]['fp'] = cm[0,1]
            tpfptnfn[threshold]['tn'] = cm[0,0]
            tpfptnfn[threshold]['fn'] = cm[1,0]
        return None, tpfptnfn
    
    def get_state_dict(self):
        return None
    
    def load_state_dict(self, state_dict):
        return


class SVMClient():
    def __init__(self, name:str, seed:int, nodes_train:str, nodes_test:str, valset_size:float, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, class_weight='balanced', random_state=42, cache_size=200, max_iter=-1, **kwargs):
        self.name = name
        
        train_df = pd.read_csv(nodes_train).drop(columns=['account', 'bank'])
        val_df = train_df.sample(frac=valset_size, random_state=seed)
        train_df = train_df.drop(val_df.index)
        test_df = pd.read_csv(nodes_test).drop(columns=['account', 'bank'])
        
        self.X_train = train_df.drop(columns=['is_sar']).to_numpy()
        self.y_train = train_df['is_sar'].to_numpy()
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = val_df.drop(columns=['is_sar']).to_numpy()
        self.X_val = scaler.transform(self.X_val)
        self.y_val = val_df['is_sar'].to_numpy()
        self.X_test = test_df.drop(columns=['is_sar']).to_numpy()
        self.X_test = scaler.transform(self.X_test)
        self.y_test = test_df['is_sar'].to_numpy()
        
        self.model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            class_weight=class_weight,
            cache_size=cache_size,
            max_iter=max_iter,
            random_state=random_state
        )
    
    def run(self, **kwargs):
        self.train()
        train_loss, train_tpfptnfn = self.evaluate(dataset='trainset')
        val_loss, val_tpfptnfn = self.evaluate(dataset='valset')
        test_loss, test_tpfptnfn = self.evaluate(dataset='testset')
        results = {0: {'train': {'loss': train_loss, 'tpfptnfn': train_tpfptnfn}, 'val': {'loss': val_loss, 'tpfptnfn': val_tpfptnfn}, 'test': {'loss': test_loss, 'tpfptnfn': test_tpfptnfn}}}
        return results
    
    def train(self):
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
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        for threshold in range(0, 101):
            cm = confusion_matrix(y, (y_pred[:,1] >= (threshold / 100)), labels=[0, 1])
            tpfptnfn[threshold]['tp'] = cm[1,1]
            tpfptnfn[threshold]['fp'] = cm[0,1]
            tpfptnfn[threshold]['tn'] = cm[0,0]
            tpfptnfn[threshold]['fn'] = cm[1,0]
        return None, tpfptnfn
    
    def get_state_dict(self):
        return None
    
    def load_state_dict(self, state_dict):
        return
    

class KNNClient():
    def __init__(self, name:str, seed:int, nodes_train:str, nodes_test:str, valset_size:float, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', n_jobs=-1, **kwargs):
        self.name = name
        
        train_df = pd.read_csv(nodes_train).drop(columns=['account', 'bank'])
        val_df = train_df.sample(frac=valset_size, random_state=seed)
        train_df = train_df.drop(val_df.index)
        test_df = pd.read_csv(nodes_test).drop(columns=['account', 'bank'])
        
        self.X_train = train_df.drop(columns=['is_sar']).to_numpy()
        self.y_train = train_df['is_sar'].to_numpy()
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = val_df.drop(columns=['is_sar']).to_numpy()
        self.X_val = scaler.transform(self.X_val)
        self.y_val = val_df['is_sar'].to_numpy()
        self.X_test = test_df.drop(columns=['is_sar']).to_numpy()
        self.X_test = scaler.transform(self.X_test)
        self.y_test = test_df['is_sar'].to_numpy()
        
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            n_jobs=n_jobs
        )
    
    def run(self, **kwargs):
        self.train()
        train_loss, train_tpfptnfn = self.evaluate(dataset='trainset')
        val_loss, val_tpfptnfn = self.evaluate(dataset='valset')
        test_loss, test_tpfptnfn = self.evaluate(dataset='testset')
        results = {0: {'train': {'loss': train_loss, 'tpfptnfn': train_tpfptnfn}, 'val': {'loss': val_loss, 'tpfptnfn': val_tpfptnfn}, 'test': {'loss': test_loss, 'tpfptnfn': test_tpfptnfn}}}
        return results
    
    def train(self):
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
        roc_auc = roc_auc_score(y, y_pred[:,1])
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        for threshold in range(0, 101):
            cm = confusion_matrix(y, (y_pred[:,1] >= (threshold / 100)), labels=[0, 1])
            tpfptnfn[threshold]['tp'] = cm[1,1]
            tpfptnfn[threshold]['fp'] = cm[0,1]
            tpfptnfn[threshold]['tn'] = cm[0,0]
            tpfptnfn[threshold]['fn'] = cm[1,0]
        return None, tpfptnfn
    
    def get_state_dict(self):
        return None
    
    def load_state_dict(self, state_dict):
        return


class MLPClient():
    def __init__(self, name:str, seed:int, device:str, nodes_train:str, nodes_test:str, valset_size:float, batch_size=64, optimizer='SGD', optimizer_params={}, criterion='ClassBalancedLoss', criterion_params={}, n_hidden_layers=2, hidden_dim=64, **kwargs):
        self.name = name
        self.device = device
        
        train_df = pd.read_csv(nodes_train).drop(columns=['account', 'bank'])
        val_df = train_df.sample(frac=valset_size, random_state=seed)
        train_df = train_df.drop(val_df.index)
        test_df = pd.read_csv(nodes_test).drop(columns=['account', 'bank'])
        
        self.trainset, self.valset, self.testset = tensordatasets(train_df, val_df, test_df, normalize=True, device=self.device)
        self.trainloader, self.valloader, self.testloader = dataloaders(self.trainset, self.valset, self.testset, batch_size)
        
        input_dim = self.trainset.tensors[0].shape[1]
        output_dim = self.trainset.tensors[1].unique().shape[0]
        self.model = MLP(input_dim=input_dim, n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim, output_dim=output_dim).to(self.device)
        
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), **optimizer_params)
        if criterion == 'ClassBalancedLoss':
            n_samples_per_classes = [sum(self.trainset.tensors[1] == 0).detach().cpu().numpy(), sum(self.trainset.tensors[1] == 1).detach().cpu().numpy()]
            self.criterion = criterions.ClassBalancedLoss(n_samples_per_classes=n_samples_per_classes, **criterion_params)
        else:
            self.criterion = getattr(torch.nn, criterion)(**criterion_params)

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
                cm = confusion_matrix(y_batch.cpu(), (y_pred[:,1] >= (threshold / 100)).to(torch.int64).cpu(), labels=[0, 1], normalize='all')
                tpfptnfn[threshold]['tp'] += cm[1,1]
                tpfptnfn[threshold]['fp'] += cm[0,1]
                tpfptnfn[threshold]['tn'] += cm[0,0]
                tpfptnfn[threshold]['fn'] += cm[1,0]
        return loss, tpfptnfn

    def evaluate(self, state_dict=None, dataset='testset'):
        if state_dict:
            self.model.load_state_dict(state_dict)
        if dataset == 'trainset':
            dataset = self.trainset
        elif dataset == 'valset':
            dataset = self.valset
        elif dataset == 'testset':
            if self.testloader == None:
                return None, None
            dataset = self.testset
        self.model.eval()
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        with torch.no_grad():
            y_pred = self.model(dataset.tensors[0])
            loss = self.criterion(y_pred, dataset.tensors[1]).item()
            for threshold in range(0, 101):
                cm = confusion_matrix(dataset.tensors[1].cpu(), (y_pred[:,1] >= (threshold / 100)).to(torch.int64).cpu(), labels=[0, 1])
                tpfptnfn[threshold]['tp'] = cm[1,1]
                tpfptnfn[threshold]['fp'] = cm[0,1]
                tpfptnfn[threshold]['tn'] = cm[0,0]
                tpfptnfn[threshold]['fn'] = cm[1,0]
        return loss, tpfptnfn
    
    def run(self, state_dict=None, n_rounds=100, eval_every=10, lr_patience=5, es_patience=15, **kwargs):
        if state_dict:
            self.model.load_state_dict(state_dict)
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        
        results_dict = {0: {}}
        loss, tpfptnfn = self.evaluate(dataset='trainset')
        results_dict[0]['train'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
        previous_train_loss = loss
        if eval_every is not None and self.valset is not None:
            loss, tpfptnfn = self.evaluate(dataset='valset')
            results_dict[0]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
            previous_val_loss = loss

        for epoch in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
            
            loss, tpfptnfn = self.train()
            results_dict[epoch] = {'train': {'loss': loss, 'tpfptnfn': tpfptnfn}}
            if loss >= previous_train_loss - 0.0005:
                lr_patience -= 1
            else:
                lr_patience = lr_patience_reset
            if lr_patience <= 0:
                tqdm.write('Decreasing learning rate.')
                decrease_lr(self.optimizer, factor=0.5)
                lr_patience = lr_patience_reset
            previous_train_loss = loss
            
            if eval_every is not None and epoch % eval_every == 0 and self.valset is not None:
                loss, tpfptnfn = self.evaluate(dataset='valset')
                results_dict[epoch]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
                if loss >= previous_val_loss - 0.0005:
                    es_patience -= eval_every
                else:
                    es_patience = es_patience_reset
                if es_patience <= 0:
                    tqdm.write('Early stopping.')
                    break
                previous_val_loss = loss
        
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


class GraphSAGEClient():
    def __init__(self, name:str, seed:int, device:str, nodes_train:str, edges_train:str, nodes_test:str, edges_test:str, valset_size:float, hidden_dim:int, optimizer:str, lr:float, weight_decay:float, criterion:str, gamma:float, **kwargs):
        self.name = name
        self.device = device
        
        train_nodes_df = pd.read_csv(nodes_train).drop(columns=['bank']).rename(columns={'account': 'node'})
        train_edges_df = pd.read_csv(edges_train)
        test_nodes_df = pd.read_csv(nodes_test).drop(columns=['bank']).rename(columns={'account': 'node'})
        test_edges_df = pd.read_csv(edges_test)
        
        self.trainset, self.testset = utils.graphdataset(train_nodes_df, train_edges_df, test_nodes_df, test_edges_df, device=device)
        self.trainset = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0)(self.trainset)
        
        input_dim = self.trainset.num_features
        output_dim = len(self.trainset.y.unique())
        self.model = GraphSAGE(input_dim=input_dim, hidden_dim=64, output_dim=output_dim).to(self.device)
        
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if criterion == 'ClassBalancedLoss':
            n_samples_per_classes = self.trainset.y.unique(return_counts=True)[1].tolist()
            self.criterion = criterions.ClassBalancedLoss(n_samples_per_classes=n_samples_per_classes, gamma=gamma)
        #elif criterion == 'NLLLoss':
        #    weight = torch.tensor([1-criterion_params['weight'], criterion_params['weight']]).to(self.device)
        #    self.criterion = getattr(torch.nn, criterion)(weight=weight)
        else:
            self.criterion = getattr(torch.nn, criterion)()
    
    def train(self, state_dict=None):
        if state_dict:
            self.model.load_state_dict(state_dict)
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(self.trainset)
        loss = self.criterion(y_pred[self.trainset.train_mask], self.trainset.y[self.trainset.train_mask])
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        tpfptnfn = {}
        for threshold in range(0, 101):
            cm = confusion_matrix(self.trainset.y[self.trainset.train_mask].cpu(), (y_pred[self.trainset.train_mask][:,1] >= (threshold / 100)).to(torch.int64).cpu(), labels=[0, 1])
            tpfptnfn[threshold] = {'tp': cm[1,1], 'fp': cm[0,1], 'tn': cm[0,0], 'fn': cm[1,0]}
        return loss, tpfptnfn
    
    def evaluate(self, state_dict=None, dataset='trainset'):
        if state_dict:
            self.model.load_state_dict(state_dict)
        if dataset == 'trainset':
            dataset = self.trainset
            mask = dataset.train_mask
        elif dataset == 'valset':
            if self.trainset.val_mask.sum() == 0:
                return None, None
            dataset = self.trainset
            mask = dataset.val_mask
        elif dataset == 'testset':
            if self.testset is None:
                return None, None
            dataset = self.testset
            mask = torch.tensor([True] * len(dataset.y))
        self.model.eval()
        tpfptnfn = {threshold: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for threshold in range(0, 101)}
        with torch.no_grad():
            y_pred = self.model(dataset)
            loss = self.criterion(y_pred[mask], dataset.y[mask]).item()
            for threshold in range(0, 101):
                cm = confusion_matrix(dataset.y[mask].cpu(), (y_pred[mask][:,1] >= (threshold / 100)).to(torch.int64).cpu(), labels=[0, 1])
                tpfptnfn[threshold]['tp'] = cm[1,1]
                tpfptnfn[threshold]['fp'] = cm[0,1]
                tpfptnfn[threshold]['tn'] = cm[0,0]
                tpfptnfn[threshold]['fn'] = cm[1,0]
        return loss, tpfptnfn
    
    def run(self, state_dict=None, n_rounds=100, eval_every=10, lr_patience=5, es_patience=15, **kwargs):
        if state_dict:
            self.model.load_state_dict(state_dict)
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        
        results_dict = {0: {}}
        loss, tpfptnfn = self.evaluate(dataset='trainset')
        results_dict[0]['train'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
        previous_train_loss = loss
        if eval_every is not None and self.trainset.val_mask.sum() > 0:
            loss, tpfptnfn = self.evaluate(dataset='valset')
            results_dict[0]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
            previous_val_loss = loss

        for epoch in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
            
            loss, tpfptnfn = self.train()
            results_dict[epoch] = {'train': {'loss': loss, 'tpfptnfn': tpfptnfn}}
            if loss >= previous_train_loss - 0.0005:
                lr_patience -= 1
            else:
                lr_patience = lr_patience_reset
            if lr_patience <= 0:
                tqdm.write('Decreasing learning rate.')
                decrease_lr(self.optimizer, factor=0.5)
                lr_patience = lr_patience_reset
            previous_train_loss = loss
            
            if eval_every is not None and epoch % eval_every == 0 and self.trainset.val_mask.sum() > 0:
                loss, tpfptnfn = self.evaluate(dataset='valset')
                results_dict[epoch]['val'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
                if loss >= previous_val_loss - 0.0005:
                    es_patience -= eval_every
                else:
                    es_patience = es_patience_reset
                if es_patience <= 0:
                    tqdm.write('Early stopping.')
                    break
                previous_val_loss = loss
        
        if eval_every is not None and self.testset is not None:
            loss, tpfptnfn = self.evaluate(dataset='testset')
            results_dict[epoch]['test'] = {'loss': loss, 'tpfptnfn': tpfptnfn}
                
        return results_dict
    
    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            state_dict[key] = value.to(self.device)
        self.model.set_state_dict(state_dict)
    
    def get_state_dict(self):
        model = self.model.get_state_dict()
        for key, value in model.items():
            model[key] = value.detach().cpu()
        return model
    
