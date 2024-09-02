import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from classifier import Classifier
from optimizer import Optimizer
from preprocess.feature_engineering import cal_features
import simulate as sim 
import utils


def main():
    
    parser = argparse.ArgumentParser(description='Automatic tuner for AMLsim parameters')
    parser.add_argument('-c', '--conf', type=str, help='Path to the config file', default='/flib/auto-aml-data-gen/param_files/10K_accts/conf.json')
    parser.add_argument('-n', '--n_trials', type=int, help='Number of optimization trials', default=10)
    parser.add_argument('-r', '--operating_recall', type=float, help='Operating recall. Value between 0.0 and 1.0.', default=0.9)
    parser.add_argument('-t', '--fpr_target', type=float, help='Target false positive rate. Value between 0.0 and 1.0', default=0.95)
    parser.add_argument('-b', '--bank', type=str, help='Bank name', default='bank')
    parser.add_argument('-w', '--num_windows', type=int, help='Number of windows used in the feature engineering', default=5)
    parser.add_argument('-l', '--window_len', type=int, help='Length of the windows used in the feature engineering', default=112)
    parser.add_argument('-o', '--overlap', type=float, help='Overlap between windows', default=0.9) 
    args = parser.parse_args()
    
    print('\n##======== Automatic tuner for AMLsim parameters ========##\n')
    print(f'config_file_path: {args.conf}')
    print(f'n_trials: {args.n_trials}')
    print(f'operating_recall: {args.operating_recall}')
    print(f'target: {args.fpr_target}')
    print(f'bank: {args.bank}')
    print(f'num_windows: {args.num_windows}')
    print(f'window_len: {args.window_len}')
    print(f'overlap: {args.overlap}\n')
    
    # find max fpr
    utils.set_same_temp_params(args.conf)
    tx_log_path = sim.run_simulation(args.conf)
    datasets = cal_features(path_to_tx_log=tx_log_path, banks=[args.bank], windows=(args.num_windows, args.window_len), overlap=args.overlap, include_edges=False)
    trainset, testset = datasets[0]
    trainset_nodes, trainset_edges = trainset
    testset_nodes, testset_edges = testset
    classifier = Classifier(dataset=(trainset_nodes, testset_nodes))
    model = classifier.train(model='GradientBoostingClassifier', tune_hyperparameters=False)
    fpr_max, importances = classifier.evaluate(operating_recall=args.operating_recall)

    optimizer = Optimizer(config_path=args.conf, target=args.fpr_target, max=fpr_max, operating_recall=args.operating_recall, model='GradientBoostingClassifier', bank=args.bank, windows=(args.num_windows, args.window_len), overlap=args.overlap)
    best_trials = optimizer.optimize(n_trials=args.n_trials)
    for trial in best_trials:
        print(f'\ntrial: {trial.number}')
        print(f'values: {trial.values}')
        for param in trial.params:
            print(f'{param}: {trial.params[param]}')
    
    return

if __name__ == '__main__':
    main()
    