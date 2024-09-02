import optuna
from simulate import create_param_files, run_simulation
from classifier import Classifier
import pandas as pd
import matplotlib.pyplot as plt
from preprocess.feature_engineering import cal_features
import json
import os

class Optimizer():
    def __init__(self, config_path, target:float, max:float=1.0, operating_recall:float=0.8, model:str='GradientBoostingClassifier', bank='bank', windows=1, overlap=0.9):
        self.config_path = config_path
        self.target = target
        self.max = max
        self.operating_recall = operating_recall
        self.model = model
        self.bank = bank
        self.windows = windows
        self.overlap = overlap
    
    def objective(self, trial:optuna.Trial):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        config['default']['mean_amount_sar'] = trial.suggest_int('mean_amount_sar', config['optimisation_bounds']['mean_amount_sar'][0], config['optimisation_bounds']['mean_amount_sar'][1])
        config['default']['std_amount_sar'] = trial.suggest_int('std_amount_sar', config['optimisation_bounds']['std_amount_sar'][0], config['optimisation_bounds']['std_amount_sar'][1])
        config['default']['mean_outcome_sar'] = trial.suggest_int('mean_outcome_sar', config['optimisation_bounds']['mean_outcome_sar'][0], config['optimisation_bounds']['mean_outcome_sar'][1])
        config['default']['std_outcome_sar'] = trial.suggest_int('std_outcome_sar', config['optimisation_bounds']['std_outcome_sar'][0], config['optimisation_bounds']['std_outcome_sar'][1])
        config['default']['prob_spend_cash'] = trial.suggest_float('prob_spend_cash', config['optimisation_bounds']['prob_spend_cash'][0], config['optimisation_bounds']['prob_spend_cash'][1])
        config['default']['mean_phone_change_frequency_sar'] = trial.suggest_int('mean_phone_change_frequency_sar', config['optimisation_bounds']['mean_phone_change_frequency_sar'][0], config['optimisation_bounds']['mean_phone_change_frequency_sar'][1])
        config['default']['std_phone_change_frequency_sar'] = trial.suggest_int('std_phone_change_frequency_sar', config['optimisation_bounds']['std_phone_change_frequency_sar'][0], config['optimisation_bounds']['std_phone_change_frequency_sar'][1])
        config['default']['mean_bank_change_frequency_sar'] = trial.suggest_int('mean_bank_change_frequency_sar', config['optimisation_bounds']['mean_bank_change_frequency_sar'][0], config['optimisation_bounds']['mean_bank_change_frequency_sar'][1])
        config['default']['std_bank_change_frequency_sar'] = trial.suggest_int('std_bank_change_frequency_sar', config['optimisation_bounds']['std_bank_change_frequency_sar'][0], config['optimisation_bounds']['std_bank_change_frequency_sar'][1])
        config['default']['prob_participate_in_multiple_sars'] = trial.suggest_float('prob_participate_in_multiple_sars', config['optimisation_bounds']['prob_participate_in_multiple_sars'][0], config['optimisation_bounds']['prob_participate_in_multiple_sars'][1])
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        tx_log_path = run_simulation(self.config_path)
        datasets = cal_features(tx_log_path, [self.bank], self.windows, self.overlap)
        trainset, testset = datasets[0]
        trainset_nodes, trainset_edges = trainset
        testset_nodes, testset_edges = testset
        classifier = Classifier(dataset=(trainset_nodes, testset_nodes))
        model = classifier.train(model=self.model, tune_hyperparameters=False)
        fpr, importances = classifier.evaluate(operating_recall=self.operating_recall)

        avg_importance = importances.mean()
        avg_importance_error = abs(importances - avg_importance)
        sum_avg_importance_error = avg_importance_error.sum()
        
        return abs(fpr/self.max-self.target), sum_avg_importance_error
    
    def optimize(self, n_trials:int=10):
        parent_dir = '/'.join(self.config_path.split('/')[:-1])
        storage = 'sqlite:///' + parent_dir + '/amlsim_study.db'
        study = optuna.create_study(storage=storage, sampler=optuna.samplers.TPESampler(), study_name='amlsim_study', directions=['minimize', 'minimize'], load_if_exists=True)
        study.optimize(self.objective, n_trials=n_trials)
        optuna.visualization.matplotlib.plot_pareto_front(study, target_names=['fpr_loss', 'importance_loss'])
        # get parent folder to self.config_path
        
        fig_path = parent_dir + '/pareto_front.png'
        plt.savefig(fig_path)
        
        log_path = parent_dir + '/log.txt'
        with open(log_path, 'w') as f:
            for trial in study.best_trials:
                f.write(f'\ntrial: {trial.number}\n')
                f.write(f'values: {trial.values}\n')
                for param in trial.params:
                    f.write(f'{param}: {trial.params[param]}\n')
        
        return study.best_trials
    





