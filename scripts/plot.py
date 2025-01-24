import argparse
import pickle
from flib.vizualize import plot_metrics

def main():
    EXPERIMENT = '3_banks_homo_mid'
    parser = argparse.ArgumentParser() # LogRegClient, DecisionTreeClient, RandomForestClient, GradientBoostingClient, SVMClient, KNNClient
    parser.add_argument('--results_files', type=str, help='Path to results file.', default=[
        f'experiments/{EXPERIMENT}/results/centralized/LogRegClient/results.pkl',
        #f'experiments/{EXPERIMENT}/results/federated/LogRegClient/results.pkl',
        #f'experiments/{EXPERIMENT}/results/isolated/LogRegClient/results.pkl',
    ])
    parser.add_argument('--metrics', nargs='+', help='Metrics to plot. Can be "loss", "accuracy", "balanced_accuracy", "recall", "precision", "f1", "roc_curve", "precision_recall_curve".', default=['loss', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'f1', 'roc_curve', 'precision_recall_curve'])
    parser.add_argument('--clients', nargs='+', help='Clients to include. If omited all clients will be included.', default=None)
    parser.add_argument('--reduction', help='Type of reduction if several clients. If "none", all clinets will be plotted individully.', default='none')
    parser.add_argument('--datasets', nargs='+', help='Datasets to include. If omited all datasets will be included.', default=['train', 'val', 'test']) # 'train', 'val', 'test'
    parser.add_argument('--formats', nargs='+', help='Formats to save plots in.', default=['png', 'csv'])
    parser.add_argument('--threshold', type=int, help='Threshold for calculating accuracy, balanced accuracy, recall and precision.', default=50)
    args = parser.parse_args()
    
    for results_file in args.results_files:
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        output_dir = results_file.replace('results.pkl', '')
        plot_metrics(data=data, metrics=args.metrics, clients=args.clients, reduction=args.reduction, datasets=args.datasets, formats=args.formats, dir=output_dir, threshold=args.threshold)

if __name__ == '__main__':
    main()