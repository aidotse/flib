import argparse
from flib.train.federated import HyperparamTuner

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--models', nargs='+', help='Types of models to train.', default=['LogisticRegressor'])
    parser.add_argument('--settings', nargs='+', help='Types of settings to use. Can be "iso", "cen" or "fed".', default=['fed'])
    parser.add_argument('--trainsets', nargs='+', help='Paths to trainsets.', default=[
        '/home/edvin/Desktop/flib/experiments/data/3_banks_homo_hard/preprocessed/a_nodes_train.csv',
        '/home/edvin/Desktop/flib/experiments/data/3_banks_homo_hard/preprocessed/b_nodes_train.csv',
        '/home/edvin/Desktop/flib/experiments/data/3_banks_homo_hard/preprocessed/c_nodes_train.csv'
    ])
    parser.add_argument('--optimizer', nargs='+', help='', default=['SGD'])
    parser.add_argument('--criterion', nargs='+', help='', default=['ClassBalancedLoss'])
    parser.add_argument('--beta', nargs='+', help='Value of beta for ClassBalancedLoss.', default=[0.9999, 0.9999999999])
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--n_rounds', type=int, help='Number of traning rounds.', default=30)
    parser.add_argument('--local_epochs', nargs='+', help='Number of local epochs at clients.', default=[1])
    parser.add_argument('--batch_size', nargs='+', help='Batch size.', default=[128, 256, 512])
    parser.add_argument('--lr', nargs='+', help='Learning rate.', default=[0.0001, 1.0])
    parser.add_argument('--n_workers', type=int, help='Number of processes.', default=3)
    parser.add_argument('--device', type=str, help='Device for computations. Can be "cpu" or cuda device, e.g. "cuda:0".', default="cuda:0")
    parser.add_argument('--results_file', type=str, default='/home/edvin/Desktop/flib/experiments/results/3_banks_homo_hard/federated/best_params.txt')
    parser.add_argument('--storage', type=str, default='sqlite:////home/edvin/Desktop/flib/experiments/results/3_banks_homo_hard/federated/study.db')
    args = parser.parse_args()
    
    args.beta = tuple(args.beta)
    args.lr = tuple(args.lr)
    
    print()
    print(f'models: {args.models}')
    print(f'settings: {args.settings}')    
    print(f'trainsets:')
    for trainset in args.trainsets:
        print(f'    {trainset}')
    print(f'optimizer: {args.optimizer}')
    print(f'criterion: {args.criterion}')
    print(f'beta: {args.beta}')
    print(f'seed: {args.seed}')
    print(f'n_rounds: {args.n_rounds}')
    print(f'local_epochs: {args.local_epochs}')
    print(f'batch_size: {args.batch_size}')
    print(f'lr: {args.lr}')
    print(f'n_workers: {args.n_workers}')
    print(f'device: {args.device}')
    print(f'results_file: {args.results_file}')
    print(f'storage: {args.storage}')
    print()
    
    for model in args.models:
        if 'cen' in args.settings:
            pass
        if 'fed' in args.settings:
            print(f'Turning hyperparameters for {model} in a federated setting.')
            hyperparamtuner = HyperparamTuner(
                seed=args.seed,
                trainsets=args.trainsets, 
                n_rounds=args.n_rounds, 
                model=model,
                optimizer=args.optimizer,
                criterion=args.criterion,
                batch_size=args.batch_size,
                n_workers=args.n_workers,
                device=args.device,
                storage=args.storage,
                results_file=args.results_file
            )
            best_params, best_value = hyperparamtuner.optimize(n_trials=20)
            print(f'Best hyperparameters: {best_params}')
            print(f'Best value: {best_value}')
            
        if 'iso' in args.settings:
            pass

if __name__ == '__main__':
    main()

