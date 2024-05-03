from simulate import init_params, create_param_files, run_simulation
from preprocess import preprocess
from classifier import Classifier
from optimizer import Optimizer
import optuna
import time


def main(n_trials:int=10, ratio=0.05, operating_recall:float=0.8, target:float=0.95):
    optimizer = Optimizer(target=target, max=0.4244, operating_recall=operating_recall, ratio=ratio)
    best_trials = optimizer.optimize(n_trials=n_trials)
    for trial in best_trials:
        print(f'\ntrial: {trial.number}')
        print(f'values: {trial.values}')
        with open('log.txt', 'a') as f:
            f.write(f'\ntrial: {trial.number}\n')
            f.write(f'values: {trial.values}\n')
        for param in trial.params:
            print(f'{param}: {trial.params[param]}')
            with open('log.txt', 'a') as f:
                f.write(f'{param}: {trial.params[param]}\n')
    return

if __name__ == '__main__':
    n_trials = 1
    ratio = 0.05 # OBS: approximate ratio of SARs in the dataset, error of about 0.02 percentage points
    operating_recall = 0.9
    target = 0.95
    
    t = time.time()
    
    main(n_trials, ratio, operating_recall, target)
    
    print(f'\nTime elapsed: {time.time()-t:.2f} seconds')