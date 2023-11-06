import torch 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import math
from sklearn.preprocessing import StandardScaler
from criterions import ClassBalancedLoss


class Client:
    def __init__(self, name, device, trainset, valset, testset, Module, Optimizer, Criterion, optimizer_params=None, criterion_params=None, lr=0.01, n_epochs=1, batch_size=64):
        self.name = name
        self.device = device

        self.x_train = trainset.iloc[:, :-1].to_numpy()
        self.y_train = trainset.iloc[:, -1].to_numpy()
        self.x_test = testset.iloc[:, :-1].to_numpy()
        self.y_test = testset.iloc[:, -1].to_numpy()
        scaler = StandardScaler().fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32).to(device)
        self.y_train = torch.tensor(self.y_train, dtype=torch.int64).to(device)
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32).to(device)
        self.y_test = torch.tensor(self.y_test, dtype=torch.int64).to(device)
        if valset:
            self.x_val = valset.iloc[:, :-1].to_numpy()
            self.y_val = valset.iloc[:, -1].to_numpy()
            self.x_val = scaler.transform(self.x_val)
            self.x_val = torch.tensor(self.x_val, dtype=torch.float32).to(device)
            self.y_val = torch.tensor(self.y_val, dtype=torch.int64).to(device)
        else:
            self.x_val = []
            self.y_val = []
        
        input_dim = 34 #self.x_train.shape[1]
        output_dim = 2 #self.y_train.unique().shape[0]
        self.module = Module(input_dim=input_dim, output_dim=output_dim).to(device)
        self.optimizer = Optimizer(self.module.parameters(), lr=lr)
        if Optimizer == torch.optim.SGD and optimizer_params:
            self.optimizer = Optimizer(self.module.parameters(), lr=lr, momentum=optimizer_params['momentum'], dampening=optimizer_params['dampening'], weight_decay=optimizer_params['weight_decay'])
        else:
            self.optimizer = Optimizer(self.module.parameters(), lr=lr)
        if Criterion == ClassBalancedLoss:
            n_samples_per_classes = [sum(self.y_train == 0).detach().cpu().numpy(), sum(self.y_train == 1).detach().cpu().numpy()]
            self.criterion = Criterion(beta=criterion_params['beta'], n_samples_per_classes=n_samples_per_classes, loss_type=criterion_params['loss_type'])
        else:
            self.criterion = Criterion()
        
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_batches = int(math.ceil(len(self.x_train) / batch_size))

    def train(self, model=None, return_predictions=False):
        if model:
            self.module.load_state_dict(model)
        self.module.train()
        losses = []
        y_pred = []
        y_true = []
        for _ in range(self.n_epochs):
            for b in range(self.n_batches):
                x_batch = self.x_train[b * self.batch_size:(b + 1) * self.batch_size]
                y_batch = self.y_train[b * self.batch_size:(b + 1) * self.batch_size]
                self.optimizer.zero_grad()
                y = self.module(x_batch)
                loss = self.criterion(y, y_batch)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                if return_predictions:
                    y_pred += y.argmax(dim=1).detach().cpu().tolist()
                    y_true += y_batch.detach().cpu().tolist()
        loss = sum(losses)/len(losses) 
        return loss, y_pred, y_true

    def validate(self, model=None, return_predictions=False):
        if model:
            self.module.load_state_dict(model)
        self.module.eval()
        losses = []
        y_pred = []
        y_true = []
        with torch.no_grad():
            y = self.module(self.x_val)
            loss = self.criterion(y, self.y_val)
            losses.append(loss.item())
            if return_predictions:
                y_pred += y.argmax(dim=1).detach().cpu().tolist()
                y_true += self.y_val.detach().cpu().tolist()
        loss = sum(losses)/len(losses) 
        return loss, y_pred, y_true

    def test(self, model=None, return_predictions=False):
        if model:
            self.module.load_state_dict(model)
        self.module.eval()
        losses = []
        y_pred = []
        y_true = []
        with torch.no_grad():
            y = self.module(self.x_test)
            loss = self.criterion(y, self.y_test)
            losses.append(loss.item())
            if return_predictions:
                y_pred += y.argmax(dim=1).detach().cpu().tolist()
                y_true += self.y_test.detach().cpu().tolist()       
        loss = sum(losses)/len(losses) 
        return loss, y_pred, y_true

    def load_model(self, model):
        for key, value in model.items():
            model[key] = value.to(self.device)
        self.module.load_state_dict(model)
    
    def model(self):
        model = self.module.state_dict()
        for key, value in model.items():
            model[key] = value.detach().cpu()
        return model

