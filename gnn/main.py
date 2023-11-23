import torch
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import networkx as nx
import matplotlib.pyplot as plt

from modules import GCN
from data import EllipticDataset

def define_gcn(trial):
    n_layers = trial.suggest_int("n_layers", 2, 5)
    hidden_dim = trial.suggest_int("hidden_dim", 2**5, 2**8, log=True)
    dropout = trial.suggest_float("dropout", 0.3, 0.7)
    return GCN(165,hidden_dim,2,n_layers,dropout)
  
def objective_gcn(trial, data, train_indices, val_indices, device):
    model = define_gcn(trial).to(device)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    t = trial.suggest_float("t", 0.2, 0.6)
    for epoch in range(100):

        model.train()
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)

        tmp = torch.nn.functional.one_hot(data.y.type(torch.long)).type(torch.float)
        loss = criterion(out[train_indices], tmp[train_indices])
        y = out.detach()[:, 1]
        y = (y > t).type(torch.long)
        f1 = f1_score(data.y.cpu()[train_indices], y.cpu()[train_indices])

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valf1 = f1_score(data.y.cpu()[val_indices], y.cpu()[val_indices])
            trial.report(valf1, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
  
    torch.save(model.state_dict(), "models/gcn-" + str(trial.number) + ".pth")
    return valf1

def eval_gcn(device):
    # load data
    elliptic_data = EllipticDataset("data/elliptic_bitcoin_dataset", val_size=0.15, test_size=0.15, seed=42)
    data, train_indices, val_indices, test_indices, train_labels, val_labels, test_labels = elliptic_data.get_data()
    
    # train and optimize hyperparameters
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective_gcn(trial, data, train_indices, val_indices, device), n_trials=100, timeout=10000,
    )

    # result of hyperparamter optimization
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # retrieve best trial from hyperparameter optimization
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("\t Trial number: ", trial.number)

    # reconstruct best trained model
    state_dict = torch.load("models/gcn-" + str(trial.number) + ".pth")
    #files.download("gcn-" + str(trial.number) + ".pth")
    model = GCN(165,trial.params["hidden_dim"],2,trial.params["n_layers"],trial.params["dropout"])
    model.load_state_dict(state_dict)

    # evaluate best trained model using test set
    model.to(device)
    model.eval()
    out = model(data)
    tmp = torch.nn.functional.one_hot(data.y.type(torch.long)).type(torch.float)
    y = out.detach()[:, 1]
    y = (y > trial.params["t"]).type(torch.long)
    f1 = f1_score(data.y.cpu()[test_indices], y.cpu()[test_indices])
    acc = accuracy_score(data.y.cpu()[test_indices], y.cpu()[test_indices])
    pre = precision_score(data.y.cpu()[test_indices], y.cpu()[test_indices])
    rec = recall_score(data.y.cpu()[test_indices], y.cpu()[test_indices])
    print("test performance:")
    print(f"\t f1: {f1}")
    print(f"\t acc: {acc}")
    print(f"\t pre: {pre}")
    print(f"\t rec: {rec}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_gcn(device)

if __name__ == "__main__":
    main()