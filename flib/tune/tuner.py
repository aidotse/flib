from flib.tune import utils
from flib.tune.classifier import Classifier
from flib.tune.optimizer import Optimizer

class DataTuner:
    def __init__(self, conf_file, generator, preprocessor, target, utility, model):
        self.conf_file = conf_file
        self.generator = generator
        self.preprocessor = preprocessor
        self.target = target
        self.utility = utility
        self.model = model
    
    def calc_min(self):
        utils.set_same_temp_params(self.conf_file)
        tx_log_file = self.generator(self.conf_file)
        dataset = self.preprocessor(tx_log_file)
        classifier = Classifier(dataset, self.conf_file.replace('conf.json', ''))
        model = classifier.train(model=self.model, tune_hyperparameters=True, n_trials=100)
        score, importances = classifier.evaluate(utility=self.utility)
        return score
    
    def __call__(self, n_trials):
        
        optimizer = Optimizer(conf_file=self.conf_file, generator=self.generator, preprocessor=self.preprocessor, target=self.target, utility=self.utility, model=self.model)
        best_trials = optimizer.optimize(n_trials=n_trials)
        for trial in best_trials:
            print(f'\ntrial: {trial.number}')
            print(f'values: {trial.values}')
            for param in trial.params:
                print(f'{param}: {trial.params[param]}')
            
        return