import optuna
from simulate import create_param_files, run_simulation
from preprocess import preprocess
from classifier import Classifier
import pandas as pd
import matplotlib.pyplot as plt


class Optimizer():
    def __init__(self, target:float, max:float=1.0, operating_recall:float=0.8, ratio:float=0.05):
        self.target = target
        self.max = max
        self.operating_recall = operating_recall
        self.ratio = ratio
    
    def objective(self, trial:optuna.Trial):
        
        params = {
            'conf': {
                "general": {
                    "random_seed": 0,
                    "simulation_name": "tmp",
                    "total_steps": 30
                },
                "default": {
                    "min_amount": 1,
                    "max_amount": 150000,
                    "mean_amount": 637,
                    "std_amount": 300,
                    "mean_amount_sar": trial.suggest_int('mean_amount_sar', 643, 643),
                    "std_amount_sar": trial.suggest_int('std_amount_sar', 300, 300),
                    "prob_income": 0.0,
                    "mean_income": 0.0,
                    "std_income": 0.0,
                    "prob_income_sar": 0.0,
                    "mean_income_sar": 0.0,
                    "std_income_sar": 0.0,
                    "mean_outcome": 500, #trial.suggest_int('mean_outcome', 400, 500),
                    "std_outcome": 100, #trial.suggest_int('std_outcome', 100, 200),
                    "mean_outcome_sar": 500, #trial.suggest_int('mean_outcome_sar', 400, 500),
                    "std_outcome_sar": 100, #trial.suggest_int('std_outcome_sar', 100, 200),
                    "prob_spend_cash": trial.suggest_float('prob_spend_cash', 0.15, 0.15),
                    "n_steps_balance_history": 7,
                    "mean_phone_change_frequency": 1460,
                    "std_phone_change_frequency": 365,
                    "mean_phone_change_frequency_sar": trial.suggest_int('mean_phone_change_frequency_sar', 1330, 1330),
                    "std_phone_change_frequency_sar": trial.suggest_int('std_phone_change_frequency_sar', 543, 543),
                    "mean_bank_change_frequency": 1460,
                    "std_bank_change_frequency": 365,
                    "mean_bank_change_frequency_sar": trial.suggest_int('mean_bank_change_frequency_sar', 1414, 1414),
                    "std_bank_change_frequency_sar": trial.suggest_int('std_bank_change_frequency_sar', 541, 541),
                    "margin_ratio": 0.1,
                    "prob_participate_in_multiple_sars": trial.suggest_float('prob_participate_in_multiple_sars', 0.06, 0.06)
                },
                "input": {
                    "directory": "paramFiles/tmp",
                    "schema": "schema.json",
                    "accounts": "accounts.csv",
                    "alert_patterns": "alertPatterns.csv",
                    "normal_models": "normalModels.csv",
                    "degree": "degree.csv",
                    "transaction_type": "transactionType.csv",
                    "is_aggregated_accounts": True
                },
                "temporal": {
                    "directory": "tmp",
                    "transactions": "transactions.csv",
                    "accounts": "accounts.csv",
                    "alert_members": "alert_members.csv",
                    "normal_models": "normal_models.csv"
                },
                "output": {
                    "directory": "outputs",
                    "transaction_log": "tx_log.csv"
                },
                "graph_generator": {
                    "degree_threshold": 1
                },
                "simulator": {
                    "transaction_limit": 1000000,
                    "transaction_interval": 7,
                    "sar_interval": 7
                },
                "scale-free": {
                    "gamma": 2.0,
                    "loc": 1.0,
                    "scale": 1.0
                }
            },
            'accounts': [
                (10000, 1000, 100000, 'SWE', 'I', 'bank'),
            ],
            'alertPatterns': [
                (int(3300*self.ratio), 'fan_out', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                #(int(3300*self.ratio), 'fan_in', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                #(int(3300*self.ratio), 'cycle', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                #(int(3300*self.ratio), 'bipartite', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                #(int(3300*self.ratio), 'stack', 2, 4, 4, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                #(int(3300*self.ratio), 'scatter_gather', 2, 5, 5, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                #(int(3300*self.ratio), 'gather_scatter', 2, 5, 5, 100, 1000, 1, 28, 'bank', True, 'CASH'),
            ],
            'normalModels': [
                (int(4000), 'single', 2, 2, 2, 1, 28),
                #(int(4000), 'fan_out', 2, 3, 3, 1, 28),
                #(int(4000), 'fan_in', 2, 3, 3, 1, 28),
                #(int(4000), 'forward', 2, 3, 3, 1, 28),
                #(int(4000), 'periodical', 2, 2, 2, 1, 28),
                #(int(4000), 'mutual', 2, 2, 2, 1, 28)
            ]
        }
        
        param_folder = '/home/edvin/Desktop/flib/AMLsim/paramFiles/tmp'
        create_param_files(params, param_folder)
        run_simulation(param_folder)
        path_to_tx_log = '/home/edvin/Desktop/flib/AMLsim/outputs/tmp/tx_log.csv'
        df = pd.read_csv(path_to_tx_log)
        df = df[(df['bankOrig']!='source') & (df['bankDest']!='sink')]
        n_false = df[df['isSAR']==0].shape[0]
        n_positiv = df[df['isSAR']==1].shape[0]
        ratio = n_positiv/(n_positiv+n_false)
        print()
        print(f'ratio: {ratio:.4f}')
        datasets = preprocess(path_to_tx_log=path_to_tx_log, banks=['bank'], split_type='spatial') # TODO: preprocess should only have path_to_tx_log as input?
        trainset, testset = datasets[0]
        columns = trainset.columns
        for column in columns:
            if 'counts_out' in column:
                count_out_column = column
                break
        avg_count = trainset[count_out_column].mean()
        print(f'avg tx count per account: {avg_count:.4f}')
        print()

        classifier = Classifier(dataset=(trainset, testset))
        model = classifier.train(model='RandomForestClassifier', tune_hyperparameters=True)
        fpr, importances = classifier.evaluate(operating_recall=self.operating_recall)

        avg_importance = importances.mean()
        avg_importance_error = abs(importances - avg_importance)
        sum_avg_importance_error = avg_importance_error.sum()
        
        return abs(fpr/self.max-self.target), sum_avg_importance_error
    
    def optimize(self, n_trials:int=10):
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), directions=['minimize', 'minimize'])
        study.optimize(self.objective, n_trials=n_trials)
        optuna.visualization.matplotlib.plot_pareto_front(study, target_names=['fpr_loss', 'importance_loss'])
        plt.savefig('pareto_front.png')
        return study.best_trials
    





