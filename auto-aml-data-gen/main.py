import simulate as sim 
from preprocess import preprocess
from classifier import Classifier
from optimizer import Optimizer
import utils
import optuna
import time
import sys
from preprocess import preprocess


def main(config_path:str, n_trials:int=10, ratio=0.05, operating_recall:float=0.8, fpr_target:float=0.95):
    print('\n##======== Automatic tuner for AMLsim parameters ========##\n')
    print(f'config_path: {config_path}')
    print(f'n_trials: {n_trials}')
    print(f'ratio: {ratio}')
    print(f'operating_recall: {operating_recall}')
    print(f'target: {fpr_target}\n')
    
    # find max fpr
    utils.set_same_temp_params(config_path)
    tx_log_path = sim.run_simulation(config_path)
    datasets = preprocess(tx_log_path, banks=['bank'], split_type='temporal', test_size=0.2, overlap=0.5)
    trainset, testset = datasets[0][0], datasets[0][1]
    classifier = Classifier(dataset=(trainset, testset))
    model = classifier.train(model='RandomForestClassifier', tune_hyperparameters=True)
    fpr, importances = classifier.evaluate(operating_recall=operating_recall)
    
    #optimizer = Optimizer(target=fpr_target, max=0.4244, operating_recall=operating_recall, ratio=ratio)
    #best_trials = optimizer.optimize(n_trials=n_trials)
    #for trial in best_trials:
    #    print(f'\ntrial: {trial.number}')
    #    print(f'values: {trial.values}')
    #    with open('log.txt', 'a') as f:
    #        f.write(f'\ntrial: {trial.number}\n')
    #        f.write(f'values: {trial.values}\n')
    #    for param in trial.params:
    #        print(f'{param}: {trial.params[param]}')
    #        with open('log.txt', 'a') as f:
    #            f.write(f'{param}: {trial.params[param]}\n')
    return

if __name__ == '__main__':
    
    # Default values
    config_path = '/home/edvin/Desktop/flib/auto-aml-data-gen/param_files/10K_accts/conf.json'
    n_trials = 1
    ratio = 0.01
    operating_recall = 0.9
    fpr_target = 0.95
    
    argv = sys.argv
    for i, arg in enumerate(argv):
        if '--config' == arg:
            config_path = argv[i+1]
        if '--n_trials' == arg:
            n_trials = int(argv[i+1])
        if '--ratio' == arg:
            ratio = float(argv[i+1])
        if '--operating_recall' == arg:
            operating_recall = float(argv[i+1])
        if '--fpr_target' == arg:
            fpr_target = float(argv[i+1])
    
    main(config_path, n_trials, ratio, operating_recall, fpr_target)
    