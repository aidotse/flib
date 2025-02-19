import torch 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve
from flib import utils
from flib.utils import tensordatasets, dataloaders, decrease_lr
from flib.train.models import LogisticRegressor, MLP, GCN, GAT, GraphSAGE
from flib.train import criterions
from flib.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import torch_geometric.transforms
from flib.utils import set_random_seed
import copy

class LogRegClient():
    def __init__(self, name:str, seed:int, device:str, nodes_train:str, nodes_test:str, valset_size:float, batch_size:int, optimizer:str, lr:float, criterion:str, **kwargs):
        self.name = name
        self.device = device
        train_df = pd.read_csv(nodes_train).drop(columns=['account', 'bank'])
        val_df = train_df.sample(frac=valset_size, random_state=seed)
        train_df = train_df.drop(val_df.index)
        test_df = pd.read_csv(nodes_test).drop(columns=['account', 'bank'])
        self.trainset, self.valset, self.testset = tensordatasets(train_df, val_df, test_df, normalize=True, device=self.device)
        y=self.trainset.tensors[1].clone().detach().cpu()
        class_counts = torch.bincount(y)
        class_weights = 1.0 / class_counts
        weights = class_weights[y]
        sampler = WeightedRandomSampler(weights, num_samples=len(y), replacement=True)
        self.trainloader, self.valloader, self.testloader = dataloaders(self.trainset, self.valset, self.testset, batch_size, sampler)
        input_dim = self.trainset.tensors[0].shape[1]
        output_dim = self.trainset.tensors[1].unique().shape[0]
        self.model = LogisticRegressor(input_dim=input_dim, output_dim=output_dim).to(self.device)
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=lr)
        self.criterion = getattr(torch.nn, criterion)()

    def train(self, state_dict=None):
        if state_dict:
            self.model.load_state_dict(state_dict)
        self.model.train()
        running_loss = 0.0
        for x_batch, y_batch in self.trainloader:
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() 
        return running_loss / len(self.trainloader)

    def evaluate(self, state_dict=None, dataset='testset'):
        if state_dict:
            self.model.load_state_dict(state_dict)
        dataset_mapping = {
            'trainset': self.trainset,
            'valset': self.valset,
            'testset': self.testset
        }
        dataset = dataset_mapping.get(dataset, self.testset)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(dataset.tensors[0])
            loss = self.criterion(y_pred, dataset.tensors[1]).item()
        return loss, y_pred.cpu().numpy(), dataset.tensors[1].cpu().numpy()
    
    def run(self, state_dict=None, n_rounds=100, eval_every=10, lr_patience=5, es_patience=15, **kwargs):
        if state_dict:
            self.model.load_state_dict(state_dict)
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        results_dict = {'train': {
            "round": [0],
            "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
            "average_precision": [average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))],
            "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
            "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
            "loss": [loss],
            "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
            "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)] 
        }}
        previous_train_loss = loss
        if eval_every is not None and self.valset is not None:
            loss, y_pred, y_true = self.evaluate(dataset='valset')
            previous_val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
            results_dict["val"] = {
                "round": [0],
                "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "average_precision": [previous_val_average_precision],
                "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "loss": [loss],
                "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)] 
            }
        for round in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
            self.train()
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            results_dict["train"]["round"].append(round)
            results_dict["train"]["accuracy"].append(accuracy_score(y_true, (y_pred[:,1] > 0.5)))
            results_dict["train"]["average_precision"].append(average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0)))
            results_dict["train"]["balanced_accuracy"].append(balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5)))
            results_dict["train"]["f1"].append(f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            results_dict["train"]["loss"].append(loss)
            results_dict["train"]["precision"].append(precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            results_dict["train"]["recall"].append(recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            if loss >= previous_train_loss:
                lr_patience -= 1
            else:
                lr_patience = lr_patience_reset
            if lr_patience <= 0:
                tqdm.write(f"Decreasing learning rate, round: {round}")
                decrease_lr(self.optimizer, factor=0.5)
                lr_patience = lr_patience_reset
            previous_train_loss = loss
            if eval_every is not None and round % eval_every == 0 and self.valset is not None:
                loss, y_pred, y_true = self.evaluate(dataset='valset')
                val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
                results_dict["val"]["round"].append(round)
                results_dict["val"]["accuracy"].append(accuracy_score(y_true, (y_pred[:,1] > 0.5)))
                results_dict["val"]["average_precision"].append(val_average_precision)
                results_dict["val"]["balanced_accuracy"].append(balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5)))
                results_dict["val"]["f1"].append(f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                results_dict["val"]["loss"].append(loss)
                results_dict["val"]["precision"].append(precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                results_dict["val"]["recall"].append(recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                if val_average_precision <= previous_val_average_precision:
                    es_patience -= eval_every
                else:
                    es_patience = es_patience_reset
                if es_patience <= 0:
                    tqdm.write(f"Early stopping, round: {round}")
                    break
                previous_val_average_precision = val_average_precision
        if eval_every is not None and self.testset is not None:
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            results_dict["train"]["precision_recall_curve"] = precision_recall_curve(y_true, y_pred[:,1], pos_label=1)
            results_dict["train"]["roc_curve"] = roc_curve(y_true, y_pred[:,1], pos_label=1)
            loss, y_pred, y_true = self.evaluate(dataset='valset')
            results_dict["val"]["precision_recall_curve"] = precision_recall_curve(y_true, y_pred[:,1], pos_label=1)
            results_dict["val"]["roc_curve"] = roc_curve(y_true, y_pred[:,1], pos_label=1)
            loss, y_pred, y_true = self.evaluate(dataset='testset')
            results_dict["test"] = {
                "round": [round],
                "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "average_precision": [average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))],
                "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "loss": [loss],
                "precision_recall_curve": precision_recall_curve(y_true, y_pred[:,1], pos_label=1),
                "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "roc_curve": roc_curve(y_true, y_pred[:,1], pos_label=1)
            }
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
    def __init__(self, name:str, seed:int, device:str, nodes_train:str, nodes_test:str, valset_size:float, batch_size:int, optimizer:str, lr:float, weight_decay:float, criterion:str, n_hidden_layers:int, hidden_dim:int, **kwargs):
        self.name = name
        self.device = device
        self.seed = seed
        set_random_seed(self.seed)
        train_df = pd.read_csv(nodes_train).drop(columns=['account', 'bank'])
        val_df = train_df.sample(frac=valset_size, random_state=seed)
        train_df = train_df.drop(val_df.index)
        test_df = pd.read_csv(nodes_test).drop(columns=['account', 'bank'])
        self.trainset, self.valset, self.testset = tensordatasets(train_df, val_df, test_df, normalize=True, device=self.device)
        y = self.trainset.tensors[1].clone().detach().cpu()
        class_counts = torch.bincount(y)
        class_weights = 1.0 / class_counts
        weights = class_weights[y]
        generator = torch.Generator(device='cpu')
        generator = generator.manual_seed(seed)
        sampler = WeightedRandomSampler(weights, num_samples=len(y), replacement=True, generator=generator)
        self.trainloader, self.valloader, self.testloader = dataloaders(self.trainset, self.valset, self.testset, batch_size, sampler, generator)
        input_dim = self.trainset.tensors[0].shape[1]
        output_dim = self.trainset.tensors[1].unique().shape[0]
        self.model = MLP(input_dim=input_dim, n_hidden_layers=n_hidden_layers, hidden_dim=hidden_dim, output_dim=output_dim).to(self.device)
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = getattr(torch.nn, criterion)()

    def train(self, state_dict=None):
        if state_dict:
            self.model.load_state_dict(state_dict)
        self.model.train()
        running_loss = 0.0
        for x_batch, y_batch in self.trainloader:
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            running_loss  += loss.item() 
        return running_loss / len(self.trainloader)

    def evaluate(self, state_dict=None, dataset='testset'):
        if state_dict:
            self.model.load_state_dict(state_dict)
        dataset_mapping = {
            'trainset': self.trainset,
            'valset': self.valset,
            'testset': self.testset
        }
        dataset = dataset_mapping.get(dataset, self.testset)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(dataset.tensors[0])
            loss = self.criterion(y_pred, dataset.tensors[1]).item()
        return loss, y_pred.cpu().numpy(), dataset.tensors[1].cpu().numpy()
    
    def run(self, state_dict=None, n_rounds=100, eval_every=10, lr_patience=5, es_patience=15, **kwargs):
        if state_dict:
            self.model.load_state_dict(state_dict)
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        results_dict = {'train': {
            "round": [0],
            "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
            "average_precision": [average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))],
            "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
            "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
            "loss": [loss],
            "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
            "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)] 
        }}
        previous_train_loss = loss
        if eval_every is not None and self.valset is not None:
            loss, y_pred, y_true = self.evaluate(dataset='valset')
            previous_val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
            results_dict["val"] = {
                "round": [0],
                "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "average_precision": [previous_val_average_precision],
                "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "loss": [loss],
                "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)] 
            }
        for round in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
            set_random_seed(self.seed+round)
            self.train()
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            results_dict["train"]["round"].append(round)
            results_dict["train"]["accuracy"].append(accuracy_score(y_true, (y_pred[:,1] > 0.5)))
            results_dict["train"]["average_precision"].append(average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0)))
            results_dict["train"]["balanced_accuracy"].append(balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5)))
            results_dict["train"]["f1"].append(f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            results_dict["train"]["loss"].append(loss)
            results_dict["train"]["precision"].append(precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            results_dict["train"]["recall"].append(recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            if loss >= previous_train_loss:
                lr_patience -= 1
            else:
                lr_patience = lr_patience_reset
            if lr_patience <= 0:
                tqdm.write(f"Decreasing learning rate, round: {round}")
                decrease_lr(self.optimizer, factor=0.5)
                lr_patience = lr_patience_reset
            previous_train_loss = loss
            if eval_every is not None and round % eval_every == 0 and self.valset is not None:
                loss, y_pred, y_true = self.evaluate(dataset='valset')
                val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
                results_dict["val"]["round"].append(round)
                results_dict["val"]["accuracy"].append(accuracy_score(y_true, (y_pred[:,1] > 0.5)))
                results_dict["val"]["average_precision"].append(val_average_precision)
                results_dict["val"]["balanced_accuracy"].append(balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5)))
                results_dict["val"]["f1"].append(f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                results_dict["val"]["loss"].append(loss)
                results_dict["val"]["precision"].append(precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                results_dict["val"]["recall"].append(recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                if val_average_precision <= previous_val_average_precision:
                    es_patience -= eval_every
                else:
                    es_patience = es_patience_reset
                if es_patience <= 0:
                    tqdm.write(f"Early stopping, round: {round}")
                    break
                previous_val_average_precision = val_average_precision
            state_dict = self.model.state_dict()
            state_dict = self._average_state_dicts([state_dict])
            self.model.load_state_dict(state_dict)
        if eval_every is not None and self.testset is not None:
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            results_dict["train"]["precision_recall_curve"] = precision_recall_curve(y_true, y_pred[:,1], pos_label=1)
            results_dict["train"]["roc_curve"] = roc_curve(y_true, y_pred[:,1], pos_label=1)
            loss, y_pred, y_true = self.evaluate(dataset='valset')
            results_dict["val"]["precision_recall_curve"] = precision_recall_curve(y_true, y_pred[:,1], pos_label=1)
            results_dict["val"]["roc_curve"] = roc_curve(y_true, y_pred[:,1], pos_label=1)
            loss, y_pred, y_true = self.evaluate(dataset='testset')
            results_dict["test"] = {
                "round": [round],
                "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "average_precision": [average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))],
                "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "loss": [loss],
                "precision_recall_curve": precision_recall_curve(y_true, y_pred[:,1], pos_label=1),
                "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "roc_curve": roc_curve(y_true, y_pred[:,1], pos_label=1)
            }
        return results_dict

    def _average_state_dicts(self, state_dicts:list, weights:list=None):
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
    
    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            state_dict[key] = value.to(self.device)
        self.model.load_state_dict(state_dict)
    
    def get_state_dict(self):
        model = self.model.state_dict()
        for key, value in model.items():
            model[key] = value.detach().cpu()
        return model


class GCNClient():
    def __init__(self, name:str, seed:int, device:str, nodes_train:str, edges_train:str, nodes_test:str, edges_test:str, valset_size:float, n_conv_layers:int, hidden_dim:int, optimizer:str, lr:float, criterion:str, **kwargs):
        
        self.name = name
        self.seed = seed
        self.device = device
        
        train_nodes_df = pd.read_csv(nodes_train).drop(columns=['bank']).rename(columns={'account': 'node'})
        train_edges_df = pd.read_csv(edges_train)
        test_nodes_df = pd.read_csv(nodes_test).drop(columns=['bank']).rename(columns={'account': 'node'})
        test_edges_df = pd.read_csv(edges_test)
        
        set_random_seed(self.seed)
        self.trainset, self.testset = utils.graphdataset(train_nodes_df, train_edges_df, test_nodes_df, test_edges_df, device=device)
        self.trainset = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=valset_size, num_test=0)(self.trainset)
        
        input_dim = self.trainset.num_features
        output_dim = len(self.trainset.y.unique())
        self.model = GCN(input_dim=input_dim, n_conv_layers=n_conv_layers, hidden_dim=hidden_dim, output_dim=output_dim).to(self.device)
        
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=lr)
        if criterion == 'ClassBalancedLoss':
            n_samples_per_classes = torch.bincount(self.trainset.y).tolist()
            self.criterion = criterions.ClassBalancedLoss(n_samples_per_classes=n_samples_per_classes, gamma=kwargs.get('gamma', 0.9))
        else:
            class_counts = torch.bincount(self.trainset.y)
            self.criterion = getattr(torch.nn, criterion)(weight = class_counts.max() / class_counts)
    
    def get_gradients(self, state_dict=None):
        if state_dict:
            self.model.load_state_dict(state_dict)
        self.model.train()
        self.model.zero_grad()
        y_pred = self.model(self.trainset)
        loss = self.criterion(y_pred[self.trainset.train_mask], self.trainset.y[self.trainset.train_mask])
        loss.backward()
        gradients = {name: param.grad.clone() for name, param in self.model.named_parameters() if param.grad is not None and "layer_norm" not in name}
        return gradients
    
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
        return loss
    
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
        with torch.no_grad():
            y_pred = self.model(dataset)
            loss = self.criterion(y_pred[mask], dataset.y[mask]).item()
        return loss, y_pred.cpu().numpy(), dataset.y.cpu().numpy()
    
    def run(self, state_dict=None, n_rounds=100, eval_every=10, lr_patience=5, es_patience=15, **kwargs):
        if state_dict:
            self.model.load_state_dict(state_dict)
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        results_dict = {'train': {
            "round": [0],
            "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
            "average_precision": [average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))],
            "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
            "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
            "loss": [loss],
            "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
            "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)] 
        }}
        previous_train_loss = loss
        if eval_every is not None and self.trainset.val_mask.sum() > 0:
            loss, y_pred, y_true = self.evaluate(dataset='valset')
            previous_val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
            results_dict["val"] = {
                "round": [0],
                "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "average_precision": [previous_val_average_precision],
                "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "loss": [loss],
                "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)] 
            }
        for round in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
            set_random_seed(self.seed+round)
            self.train()
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            results_dict["train"]["round"].append(round)
            results_dict["train"]["accuracy"].append(accuracy_score(y_true, (y_pred[:,1] > 0.5)))
            results_dict["train"]["average_precision"].append(average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0)))
            results_dict["train"]["balanced_accuracy"].append(balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5)))
            results_dict["train"]["f1"].append(f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            results_dict["train"]["loss"].append(loss)
            results_dict["train"]["precision"].append(precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            results_dict["train"]["recall"].append(recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            if loss >= previous_train_loss:
                lr_patience -= 1
            else:
                lr_patience = lr_patience_reset
            if lr_patience <= 0:
                tqdm.write(f"Decreasing learning rate, round: {round}")
                decrease_lr(self.optimizer, factor=0.5)
                lr_patience = lr_patience_reset
            previous_train_loss = loss
            if eval_every is not None and round % eval_every == 0 and self.trainset.val_mask.sum() > 0:
                loss, y_pred, y_true = self.evaluate(dataset='valset')
                val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
                results_dict["val"]["round"].append(round)
                results_dict["val"]["accuracy"].append(accuracy_score(y_true, (y_pred[:,1] > 0.5)))
                results_dict["val"]["average_precision"].append(val_average_precision)
                results_dict["val"]["balanced_accuracy"].append(balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5)))
                results_dict["val"]["f1"].append(f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                results_dict["val"]["loss"].append(loss)
                results_dict["val"]["precision"].append(precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                results_dict["val"]["recall"].append(recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                if val_average_precision <= previous_val_average_precision:
                    es_patience -= eval_every
                else:
                    es_patience = es_patience_reset
                if es_patience <= 0:
                    tqdm.write(f"Early stopping, round: {round}")
                    break
                previous_val_average_precision = val_average_precision
        if eval_every is not None and self.testset is not None:
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            results_dict["train"]["precision_recall_curve"] = precision_recall_curve(y_true, y_pred[:,1], pos_label=1)
            results_dict["train"]["roc_curve"] = roc_curve(y_true, y_pred[:,1], pos_label=1)
            loss, y_pred, y_true = self.evaluate(dataset='valset')
            results_dict["val"]["precision_recall_curve"] = precision_recall_curve(y_true, y_pred[:,1], pos_label=1)
            results_dict["val"]["roc_curve"] = roc_curve(y_true, y_pred[:,1], pos_label=1)
            loss, y_pred, y_true = self.evaluate(dataset='testset')
            results_dict["test"] = {
                "round": [round],
                "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "average_precision": [average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))],
                "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "loss": [loss],
                "precision_recall_curve": precision_recall_curve(y_true, y_pred[:,1], pos_label=1),
                "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "roc_curve": roc_curve(y_true, y_pred[:,1], pos_label=1)
            }
        return results_dict
    
    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            state_dict[key] = value.clone().detach().to(self.device)
        self.model.set_state_dict(state_dict)
    
    def get_state_dict(self):
        model = self.model.get_state_dict()
        for key, value in model.items():
            model[key] = value.detach().cpu()
        return model


class GATClient():
    def __init__(self, name:str, seed:int, device:str, nodes_train:str, edges_train:str, nodes_test:str, edges_test:str, valset_size:float, n_conv_layers:int, hidden_dim:int, optimizer:str, lr:float, criterion:str, **kwargs):
        self.name = name
        self.device = device
        
        train_nodes_df = pd.read_csv(nodes_train).drop(columns=['bank']).rename(columns={'account': 'node'})
        train_edges_df = pd.read_csv(edges_train)
        test_nodes_df = pd.read_csv(nodes_test).drop(columns=['bank']).rename(columns={'account': 'node'})
        test_edges_df = pd.read_csv(edges_test)
        
        self.trainset, self.testset = utils.graphdataset(train_nodes_df, train_edges_df, test_nodes_df, test_edges_df, device=device)
        self.trainset = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=valset_size, num_test=0)(self.trainset)
        
        input_dim = self.trainset.num_features
        output_dim = len(self.trainset.y.unique())
        self.model = GAT(input_dim=input_dim, n_conv_layers=n_conv_layers, hidden_dim=hidden_dim, output_dim=output_dim).to(self.device)
        
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=lr)
        if criterion == 'ClassBalancedLoss':
            n_samples_per_classes = torch.bincount(self.trainset.y).tolist()
            self.criterion = criterions.ClassBalancedLoss(n_samples_per_classes=n_samples_per_classes, gamma=kwargs.get('gamma', 0.9))
        else:
            class_counts = torch.bincount(self.trainset.y)
            self.criterion = getattr(torch.nn, criterion)(weight = class_counts.max() / class_counts)
    
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
        return loss
    
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
        with torch.no_grad():
            y_pred = self.model(dataset)
            loss = self.criterion(y_pred[mask], dataset.y[mask]).item()
        return loss, y_pred.cpu().numpy(), dataset.y.cpu().numpy()
    
    def run(self, state_dict=None, n_rounds=100, eval_every=10, lr_patience=5, es_patience=15, **kwargs):
        if state_dict:
            self.model.load_state_dict(state_dict)
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        results_dict = {'train': {
            "round": [0],
            "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
            "average_precision": [average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))],
            "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
            "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
            "loss": [loss],
            "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
            "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)] 
        }}
        previous_train_loss = loss
        if eval_every is not None and self.trainset.val_mask.sum() > 0:
            loss, y_pred, y_true = self.evaluate(dataset='valset')
            previous_val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
            results_dict["val"] = {
                "round": [0],
                "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "average_precision": [previous_val_average_precision],
                "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "loss": [loss],
                "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)] 
            }
        for round in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
            self.train()
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            results_dict["train"]["round"].append(round)
            results_dict["train"]["accuracy"].append(accuracy_score(y_true, (y_pred[:,1] > 0.5)))
            results_dict["train"]["average_precision"].append(average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0)))
            results_dict["train"]["balanced_accuracy"].append(balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5)))
            results_dict["train"]["f1"].append(f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            results_dict["train"]["loss"].append(loss)
            results_dict["train"]["precision"].append(precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            results_dict["train"]["recall"].append(recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            if loss >= previous_train_loss:
                lr_patience -= 1
            else:
                lr_patience = lr_patience_reset
            if lr_patience <= 0:
                tqdm.write(f"Decreasing learning rate, round: {round}")
                decrease_lr(self.optimizer, factor=0.5)
                lr_patience = lr_patience_reset
            previous_train_loss = loss
            if eval_every is not None and round % eval_every == 0 and self.trainset.val_mask.sum() > 0:
                loss, y_pred, y_true = self.evaluate(dataset='valset')
                val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
                results_dict["val"]["round"].append(round)
                results_dict["val"]["accuracy"].append(accuracy_score(y_true, (y_pred[:,1] > 0.5)))
                results_dict["val"]["average_precision"].append(val_average_precision)
                results_dict["val"]["balanced_accuracy"].append(balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5)))
                results_dict["val"]["f1"].append(f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                results_dict["val"]["loss"].append(loss)
                results_dict["val"]["precision"].append(precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                results_dict["val"]["recall"].append(recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                if val_average_precision <= previous_val_average_precision:
                    es_patience -= eval_every
                else:
                    es_patience = es_patience_reset
                if es_patience <= 0:
                    tqdm.write(f"Early stopping, round: {round}")
                    break
                previous_val_average_precision = val_average_precision
        if eval_every is not None and self.testset is not None:
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            results_dict["train"]["precision_recall_curve"] = precision_recall_curve(y_true, y_pred[:,1], pos_label=1)
            results_dict["train"]["roc_curve"] = roc_curve(y_true, y_pred[:,1], pos_label=1)
            loss, y_pred, y_true = self.evaluate(dataset='valset')
            results_dict["val"]["precision_recall_curve"] = precision_recall_curve(y_true, y_pred[:,1], pos_label=1)
            results_dict["val"]["roc_curve"] = roc_curve(y_true, y_pred[:,1], pos_label=1)
            loss, y_pred, y_true = self.evaluate(dataset='testset')
            results_dict["test"] = {
                "round": [round],
                "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "average_precision": [average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))],
                "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "loss": [loss],
                "precision_recall_curve": precision_recall_curve(y_true, y_pred[:,1], pos_label=1),
                "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "roc_curve": roc_curve(y_true, y_pred[:,1], pos_label=1)
            }
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


class GraphSAGEClient():
    def __init__(self, name:str, seed:int, device:str, nodes_train:str, edges_train:str, nodes_test:str, edges_test:str, valset_size:float, n_conv_layers:int, hidden_dim:int, optimizer:str, lr:float, criterion:str, **kwargs):
        self.name = name
        self.device = device
        
        train_nodes_df = pd.read_csv(nodes_train).drop(columns=['bank']).rename(columns={'account': 'node'})
        train_edges_df = pd.read_csv(edges_train)
        test_nodes_df = pd.read_csv(nodes_test).drop(columns=['bank']).rename(columns={'account': 'node'})
        test_edges_df = pd.read_csv(edges_test)
        
        self.trainset, self.testset = utils.graphdataset(train_nodes_df, train_edges_df, test_nodes_df, test_edges_df, device=device)
        self.trainset = torch_geometric.transforms.RandomNodeSplit(split='train_rest', num_val=valset_size, num_test=0)(self.trainset)
        
        input_dim = self.trainset.num_features
        output_dim = len(self.trainset.y.unique())
        self.model = GraphSAGE(input_dim=input_dim, n_conv_layers=n_conv_layers, hidden_dim=hidden_dim, output_dim=output_dim).to(self.device)
        
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), lr=lr)
        if criterion == 'ClassBalancedLoss':
            n_samples_per_classes = torch.bincount(self.trainset.y).tolist()
            self.criterion = criterions.ClassBalancedLoss(n_samples_per_classes=n_samples_per_classes, gamma=kwargs.get('gamma', 0.9))
        else:
            class_counts = torch.bincount(self.trainset.y)
            self.criterion = getattr(torch.nn, criterion)(weight = class_counts.max() / class_counts)
    
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
        return loss
    
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
        with torch.no_grad():
            y_pred = self.model(dataset)
            loss = self.criterion(y_pred[mask], dataset.y[mask]).item()
        return loss, y_pred.cpu().numpy(), dataset.y.cpu().numpy()
    
    def run(self, state_dict=None, n_rounds=100, eval_every=10, lr_patience=5, es_patience=15, **kwargs):
        if state_dict:
            self.model.load_state_dict(state_dict)
        lr_patience_reset = lr_patience
        es_patience_reset = es_patience
        loss, y_pred, y_true = self.evaluate(dataset='trainset')
        results_dict = {'train': {
            "round": [0],
            "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
            "average_precision": [average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))],
            "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
            "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
            "loss": [loss],
            "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
            "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)] 
        }}
        previous_train_loss = loss
        if eval_every is not None and self.trainset.val_mask.sum() > 0:
            loss, y_pred, y_true = self.evaluate(dataset='valset')
            previous_val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
            results_dict["val"] = {
                "round": [0],
                "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "average_precision": [previous_val_average_precision],
                "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "loss": [loss],
                "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)] 
            }
        for round in tqdm(range(1, n_rounds+1), desc='progress', leave=False):
            self.train()
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            results_dict["train"]["round"].append(round)
            results_dict["train"]["accuracy"].append(accuracy_score(y_true, (y_pred[:,1] > 0.5)))
            results_dict["train"]["average_precision"].append(average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0)))
            results_dict["train"]["balanced_accuracy"].append(balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5)))
            results_dict["train"]["f1"].append(f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            results_dict["train"]["loss"].append(loss)
            results_dict["train"]["precision"].append(precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            results_dict["train"]["recall"].append(recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
            if loss >= previous_train_loss:
                lr_patience -= 1
            else:
                lr_patience = lr_patience_reset
            if lr_patience <= 0:
                tqdm.write(f"Decreasing learning rate, round: {round}")
                decrease_lr(self.optimizer, factor=0.5)
                lr_patience = lr_patience_reset
            previous_train_loss = loss
            if eval_every is not None and round % eval_every == 0 and self.trainset.val_mask.sum() > 0:
                loss, y_pred, y_true = self.evaluate(dataset='valset')
                val_average_precision = average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))
                results_dict["val"]["round"].append(round)
                results_dict["val"]["accuracy"].append(accuracy_score(y_true, (y_pred[:,1] > 0.5)))
                results_dict["val"]["average_precision"].append(val_average_precision)
                results_dict["val"]["balanced_accuracy"].append(balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5)))
                results_dict["val"]["f1"].append(f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                results_dict["val"]["loss"].append(loss)
                results_dict["val"]["precision"].append(precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                results_dict["val"]["recall"].append(recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0))
                if val_average_precision <= previous_val_average_precision:
                    es_patience -= eval_every
                else:
                    es_patience = es_patience_reset
                if es_patience <= 0:
                    tqdm.write(f"Early stopping, round: {round}")
                    break
                previous_val_average_precision = val_average_precision
        if eval_every is not None and self.testset is not None:
            loss, y_pred, y_true = self.evaluate(dataset='trainset')
            results_dict["train"]["precision_recall_curve"] = precision_recall_curve(y_true, y_pred[:,1], pos_label=1)
            results_dict["train"]["roc_curve"] = roc_curve(y_true, y_pred[:,1], pos_label=1)
            loss, y_pred, y_true = self.evaluate(dataset='valset')
            results_dict["val"]["precision_recall_curve"] = precision_recall_curve(y_true, y_pred[:,1], pos_label=1)
            results_dict["val"]["roc_curve"] = roc_curve(y_true, y_pred[:,1], pos_label=1)
            loss, y_pred, y_true = self.evaluate(dataset='testset')
            results_dict["test"] = {
                "round": [round],
                "accuracy": [accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "average_precision": [average_precision_score(y_true, y_pred[:,1], recall_span=(0.6, 1.0))],
                "balanced_accuracy": [balanced_accuracy_score(y_true, (y_pred[:,1] > 0.5))],
                "f1": [f1_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "loss": [loss],
                "precision_recall_curve": precision_recall_curve(y_true, y_pred[:,1], pos_label=1),
                "precision": [precision_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "recall": [recall_score(y_true, (y_pred[:,1] > 0.5), pos_label=1, zero_division=0.0)],
                "roc_curve": roc_curve(y_true, y_pred[:,1], pos_label=1)
            }
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
