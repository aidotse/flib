import torch 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import math
from sklearn.preprocessing import StandardScaler

class Client:
    def __init__(self, name, device, trainset, valset, testset, Module, Optimizer, Criterion, lr=0.01, n_epochs=1, batch_size=64):
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
        
        input_dim = self.x_train.shape[1]
        output_dim = self.y_train.unique().shape[0]
        self.module = Module(input_dim=input_dim, output_dim=output_dim).to(device)
        self.optimizer = Optimizer(self.module.parameters(), lr=lr)
        self.criterion = Criterion()

        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_batches = int(math.ceil(len(self.x_train) / batch_size))

    def train(self, model=None, return_metrics=False):
        if model:
            self.module.load_state_dict(model)
        self.module.train()
        losses = []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        cf_matrices = []
        for _ in range(self.n_epochs):
            for b in range(self.n_batches):
                x_batch = self.x_train[b * self.batch_size:(b + 1) * self.batch_size]
                y_batch = self.y_train[b * self.batch_size:(b + 1) * self.batch_size]
                self.optimizer.zero_grad()
                y_pred = self.module(x_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                y_batch = y_batch.detach().cpu()
                y_pred = y_pred.argmax(dim=1).detach().cpu()
                if return_metrics:
                    losses.append(loss.item())
                    accuracies.append(accuracy_score(y_true=y_batch, y_pred=y_pred))
                    precisions.append(precision_score(y_true=y_batch, y_pred=y_pred, zero_division=0, average='macro'))
                    recalls.append(recall_score(y_true=y_batch, y_pred=y_pred, zero_division=0, average='macro'))
                    f1s.append(f1_score(y_true=y_batch, y_pred=y_pred, zero_division=0, average='macro'))
                    cf_matrices.append(confusion_matrix(y_true=y_batch, y_pred=y_pred, labels=[0, 1]))
                else:
                    losses.append(loss.item())
                    accuracies.append(accuracy_score(y_true=y_batch, y_pred=y_pred))
        loss = sum(losses)/len(losses) 
        accuracy = sum(accuracies)/len(accuracies)
        if return_metrics:
            precision = sum(precisions)/len(precisions)
            recall = sum(recalls)/len(recalls)
            f1 = sum(f1s)/len(f1s)
            cf_matrix = sum(cf_matrices)
            return loss, accuracy, precision, recall, f1, cf_matrix
        else:
            return loss, accuracy

    def validate(self, model=None, return_metrics=False):
        if model:
            self.module.load_state_dict(model)
        self.module.eval()
        losses = []
        accuracies = [] 
        precisions = []
        recalls = []
        f1s = []
        cf_matrices = []
        with torch.no_grad():
            y_pred = self.module(self.x_val)
            loss = self.criterion(y_pred, self.y_val)
            y_val = self.y_val.detach().cpu()
            y_pred = y_pred.argmax(dim=1).detach().cpu()
            losses.append(loss.item())
            accuracies.append(accuracy_score(y_true=y_val, y_pred=y_pred))
            if return_metrics:
                precisions.append(precision_score(y_true=y_val, y_pred=y_pred, zero_division=0, average='macro'))
                recalls.append(recall_score(y_true=y_val, y_pred=y_pred, zero_division=0, average='macro'))
                f1s.append(f1_score(y_true=y_val, y_pred=y_pred, zero_division=0, average='macro'))
                cf_matrices.append(confusion_matrix(y_true=y_val, y_pred=y_pred, labels=[0, 1]))
        loss = sum(losses)/len(losses) 
        accuracy = sum(accuracies)/len(accuracies)
        if return_metrics:
            loss = sum(losses)/len(losses) 
            accuracy = sum(accuracies)/len(accuracies)
            precision = sum(precisions)/len(precisions)
            recall = sum(recalls)/len(recalls)
            f1 = sum(f1s)/len(f1s)
            cf_matrix = sum(cf_matrices)
            return loss, accuracy, precision, recall, f1, cf_matrix
        else:
            return loss, accuracy

    def test(self, model=None, return_metrics=False):
        if model:
            self.module.load_state_dict(model)
        self.module.eval()
        losses = []
        accuracies = [] 
        precisions = []
        recalls = []
        f1s = []
        cf_matrices = []
        with torch.no_grad():
            y_pred = self.module(self.x_test)
            loss = self.criterion(y_pred, self.y_test)
            y_test = self.y_test.detach().cpu()
            y_pred = y_pred.argmax(dim=1).detach().cpu()
            losses.append(loss.item())
            accuracies.append(accuracy_score(y_true=y_test, y_pred=y_pred))
            if return_metrics:
                precisions.append(precision_score(y_true=y_test, y_pred=y_pred, zero_division=0, average='macro'))
                recalls.append(recall_score(y_true=y_test, y_pred=y_pred, zero_division=0, average='macro'))
                f1s.append(f1_score(y_true=y_test, y_pred=y_pred, zero_division=0, average='macro'))
                cf_matrices.append(confusion_matrix(y_true=y_test, y_pred=y_pred, labels=[0, 1]))
        loss = sum(losses)/len(losses) 
        accuracy = sum(accuracies)/len(accuracies)
        if return_metrics:
            precision = sum(precisions)/len(precisions)
            recall = sum(recalls)/len(recalls)
            f1 = sum(f1s)/len(f1s)
            cf_matrix = sum(cf_matrices)
            return loss, accuracy, precision, recall, f1, cf_matrix
        else:
            return loss, accuracy

    def load_model(self, model):
        for key, value in model.items():
            model[key] = value.to(self.device)
        self.module.load_state_dict(model)
    
    def model(self):
        model = self.module.state_dict()
        for key, value in model.items():
            model[key] = value.detach().cpu()
        return model

