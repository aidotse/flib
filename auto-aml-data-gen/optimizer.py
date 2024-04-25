import optuna
from simulate import create_param_files, run_simulation
from preprocess import preprocess
from classifier import Classifier



class Optimizer():
    def __init__(self, target:float, max:float=1.0, operating_recall:float=0.8):
        self.target = target
        self.max = max
        self.operating_recall = operating_recall
    
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
                    "mean_amount": 500, #trial.suggest_int('mean_amount', 1, 1500),
                    "std_amount": 200, #trial.suggest_int('std_amount', 100, 500),
                    "mean_amount_sar": trial.suggest_int('mean_amount_sar', 500, 1000),
                    "std_amount_sar": trial.suggest_int('std_amount_sar', 200, 500),
                    "prob_income": 0.0,
                    "mean_income": 0.0,
                    "std_income": 0.0,
                    "prob_income_sar": 0.0,
                    "mean_income_sar": 0.0,
                    "std_income_sar": 0.0,
                    "mean_outcome": 500, #trial.suggest_int('mean_outcome', 1, 1500),
                    "std_outcome": 200, #trial.suggest_int('std_outcome', 1, 1500),
                    "mean_outcome_sar": 500, #trial.suggest_int('mean_outcome_sar', 1, 1500),
                    "std_outcome_sar": 200, #trial.suggest_int('std_outcome_sar', 1, 1500),
                    "prob_spend_cash": trial.suggest_float('prob_spend_cash', 0.0, 1.0),
                    "n_steps_balance_history": 7, #trial.suggest_int('n_steps_balance_history', 1, 30),
                    "mean_phone_change_frequency": 730, #trial.suggest_int('mean_phone_change_frequency', 1, 1500),
                    "std_phone_change_frequency": 365, #trial.suggest_int('std_phone_change_frequency', 1, 1500),
                    "mean_phone_change_frequency_sar": trial.suggest_int('mean_phone_change_frequency_sar', 1, 1500),
                    "std_phone_change_frequency_sar": trial.suggest_int('std_phone_change_frequency_sar', 1, 1500),
                    "mean_bank_change_frequency": 1460, #trial.suggest_int('mean_bank_change_frequency', 1, 1500),
                    "std_bank_change_frequency": trial.suggest_int('std_bank_change_frequency', 1, 1500),
                    "mean_bank_change_frequency_sar": trial.suggest_int('mean_bank_change_frequency_sar', 1, 1500),
                    "std_bank_change_frequency_sar": trial.suggest_int('std_bank_change_frequency_sar', 1, 1500),
                    "margin_ratio": 0.1,
                    "prob_participate_in_multiple_sars": trial.suggest_float('prob_participate_in_multiple_sars', 0.0, 1.0)
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
                    "transaction_limit": 100000,
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
                (40, 'fan_out', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                (40, 'fan_in', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                (40, 'cycle', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                (40, 'bipartite', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                (40, 'stack', 2, 4, 4, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                (40, 'scatter_gather', 2, 5, 5, 100, 1000, 1, 28, 'bank', True, 'CASH'),
                (40, 'gather_scatter', 2, 5, 5, 100, 1000, 1, 28, 'bank', True, 'CASH'),
            ],
            'normalModels': [
                (10000, 'single', 2, 2, 2, 1, 28),
                (10000, 'fan_out', 2, 3, 3, 1, 28),
                (10000, 'fan_in', 2, 3, 3, 1, 28),
                (10000, 'forward', 2, 3, 3, 1, 28),
                (10000, 'periodical', 2, 2, 2, 1, 28),
                (10000, 'mutual', 2, 2, 2, 1, 28)
            ]
        }
        
        param_folder = '/home/edvin/Desktop/flib/AMLsim/paramFiles/tmp'
        create_param_files(params, param_folder)
        run_simulation(param_folder)
        path_to_tx_log = '/home/edvin/Desktop/flib/AMLsim/outputs/tmp/tx_log.csv'
        datasets = preprocess(path_to_tx_log, ['bank'], 0.0)
        trainset, testset = datasets[0]
        
        #print(f'trainset - n samples: {trainset.shape[0]}, label ratio: {trainset[trainset["is_sar"]==1].shape[0]/trainset[trainset["is_sar"]==0].shape[0]:.4f}')
        #print(f'testset - n samples: {testset.shape[0]}, label ratio: {testset[testset["is_sar"]==1].shape[0]/testset[testset["is_sar"]==0].shape[0]:.4f}')

        classifier = Classifier(dataset=(trainset, testset))
        model = classifier.train(model='RandomForestClassifier', tune_hyperparameters=True)
        fpr, importances = classifier.evaluate(operating_recall=self.operating_recall)

        avg_importance = importances.mean()
        avg_importance_error = abs(importances - avg_importance)
        sum_avg_importance_error = avg_importance_error.sum()
        
        return abs(fpr/self.max-self.target), sum_avg_importance_error
    
    def optimize(self, n_trials:int=10):
        study = optuna.create_study(directions=['minimize', 'minimize'])
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_trials
    





