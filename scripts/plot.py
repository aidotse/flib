import argparse
import pickle
from flib.vizualize import plot_metrics

def main():
    parser = argparse.ArgumentParser() # LogRegClient, DecisionTreeClient, RandomForestClient, GradientBoostingClient, SVMClient, KNNClient
    parser.add_argument('--results_files', type=str, help='Path to results file.', default=[
        'experiments/3_banks_homo_mid/results/centralized/LogRegClient/results.pkl',
        'experiments/3_banks_homo_mid/results/federated/LogRegClient/results.pkl',
        'experiments/3_banks_homo_mid/results/isolated/LogRegClient/results.pkl',
    ])
    parser.add_argument('--metrics', nargs='+', help='Metrics to plot. Can be "loss", "accuracy", "balanced_accuracy", "recall", "precision", "average_precision", "f1", "roc_curve", "precision_recall_curve".', default=['loss', 'accuracy', 'balanced_accuracy', 'recall', 'precision', 'average_precision', 'f1', 'roc_curve', 'precision_recall_curve'])
    parser.add_argument('--clients', nargs='+', help='Clients to include. If omited all clients will be included.', default=None)
    parser.add_argument('--reduction', help='Type of reduction if several clients. If "none", all clinets will be plotted individully.', default='mean')
    parser.add_argument('--datasets', nargs='+', help='Datasets to include. If omited all datasets will be included.', default=['train', 'val', 'test']) # 'train', 'val', 'test'
    parser.add_argument('--formats', nargs='+', help='Formats to save plots in.', default=['png', 'csv'])
    parser.add_argument('--threshold', type=int, help='Threshold for calculating accuracy, balanced accuracy, recall and precision.', default=50)
    parser.add_argument('--recall_span', nargs='+', help='Recall span for caculating average precision.', default=(0.6, 0.8))
    args = parser.parse_args()
    
    for results_file in args.results_files:
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        output_dir = results_file.replace('results.pkl', '')
        plot_metrics(data=data, metrics=args.metrics, clients=args.clients, reduction=args.reduction, datasets=args.datasets, formats=args.formats, dir=output_dir, threshold=args.threshold, recall_span=args.recall_span)

if __name__ == '__main__':
    main()