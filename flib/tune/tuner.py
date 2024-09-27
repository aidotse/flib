from flib.tune import utils
from flib.tune.classifier import Classifier
from flib.tune.optimizer import Optimizer

class DataTuner:
    def __init__(self, conf_file, generator, preprocessor, operating_recall, fpr_target):
        self.conf_file = conf_file
        self.generator = generator
        self.preprocessor = preprocessor
        self.operating_recall = operating_recall
        self.fpr_target = fpr_target
        self.fpr_max = None
    
    def get_fpr_max(self):
        return self.fpr_max
    
    def set_fpr_max(self, fpr_max):
        self.fpr_max = fpr_max
        return self.fpr_max
    
    def __call__(self, n_trials):
        if self.fpr_max is None: 
            # find max fpr
            utils.set_same_temp_params(self.conf_file)
            tx_log_file = self.generator(self.conf_file)
            dataset = self.preprocessor(tx_log_file)
            classifier = Classifier(dataset)
            model = classifier.train(model='DecisionTreeClassifier', tune_hyperparameters=False)
            self.fpr_max, importances = classifier.evaluate(operating_recall=self.operating_recall)
            print(f'fpr_max: {self.fpr_max}')
        
        optimizer = Optimizer(conf_file=self.conf_file, generator=self.generator, preprocessor=self.preprocessor, target=self.fpr_target, max=self.fpr_max, operating_recall=self.operating_recall, model='DecisionTreeClassifier')
        best_trials = optimizer.optimize(n_trials=n_trials)
        for trial in best_trials:
            print(f'\ntrial: {trial.number}')
            print(f'values: {trial.values}')
            for param in trial.params:
                print(f'{param}: {trial.params[param]}')
            
        return