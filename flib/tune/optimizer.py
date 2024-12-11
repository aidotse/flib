import optuna
from flib.tune.classifier import Classifier # TODO: classifiers should be in flib.models
import matplotlib.pyplot as plt
import json

class Optimizer():
    def __init__(self, conf_file, generator, preprocessor, target:float, utility:str, model:str='DecisionTreeClassifier', bank=None):
        self.conf_file = conf_file
        self.generator = generator
        self.preprocessor = preprocessor
        self.target = target
        self.utility = utility
        self.model = model
        self.bank = bank
    
    def objective(self, trial:optuna.Trial):
        with open(self.conf_file, 'r') as f:
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
        
        with open(self.conf_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        tx_log_file = self.generator(self.conf_file)
        dataset = self.preprocessor(tx_log_file)
        
        if dataset[0]['is_sar'].nunique() != 2:
            print(f'\nWarning: {dataset[0].loc[0, "bank"]} trainset has only one class {dataset[0]["is_sar"].unique()}\n')
            return 1.0, 5.0
        if dataset[1]['is_sar'].nunique() != 2:
            print(f'\nWarning: {dataset[1].loc[0, "bank"]} testset has only one class {dataset[1]["is_sar"].unique()}\n')
            return 1.0, 5.0
        
        classifier = Classifier(dataset, results_dir=self.conf_file.replace('conf.json', ''))
        model = classifier.train(model=self.model, tune_hyperparameters=True, n_trials=100)
        score, importances = classifier.evaluate(utility=self.utility)

        avg_importance = importances.mean()
        avg_importance_error = abs(avg_importance - importances)
        sum_avg_importance_error = avg_importance_error.sum()
        
        return abs(score-self.target), sum_avg_importance_error
    
    def optimize(self, n_trials:int=10):
        parent_dir = '/'.join(self.conf_file.split('/')[:-1])
        storage = 'sqlite:///' + parent_dir + '/amlsim_study.db'
        study = optuna.create_study(storage=storage, sampler=optuna.samplers.TPESampler(), study_name='amlsim_study', directions=['minimize', 'minimize'], load_if_exists=True)
        study.optimize(self.objective, n_trials=n_trials)
        optuna.visualization.matplotlib.plot_pareto_front(study, target_names=[self.utility+'_loss', 'importance_loss'])
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
    





