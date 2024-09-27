import numpy as np
import matplotlib.pyplot as plt
import os

def accuracy(data):
    tp, fp, tn, fn = data['tp'], data['fp'], data['tn'], data['fn']
    return (tp+tn)/(tp+fp+tn+fn)

def balanced_accuracy(data):
    tp, fp, tn, fn = data['tp'], data['fp'], data['tn'], data['fn']
    return 0.5*(tp/(tp+fn) + tn/(tn+fp))

def recall(data):
    tp, fp, tn, fn = data['tp'], data['fp'], data['tn'], data['fn']
    return tp/(tp+fn)

def precision(data):
    tp, fp, tn, fn = data['tp'], data['fp'], data['tn'], data['fn']
    if tp+fp == 0.0:
        return 0.0
    return tp/(tp+fp)

def f1(data):
    tp, fp, tn, fn = data['tp'], data['fp'], data['tn'], data['fn']
    return 2*tp/(2*tp+fp+fn)
    
def roc_curve(data):
    tprs = [0]
    fprs = [0]
    for threshold in data:
        tp, fp, tn, fn = data[threshold]['tp'], data[threshold]['fp'], data[threshold]['tn'], data[threshold]['fn']
        tprs.append(tp/(tp+fn))
        fprs.append(fp/(fp+tn))
    tprs.append(1.0)
    fprs.append(1.0)
    return tprs, fprs

def precision_recall_curve(data):
    recalls = [0.0]
    precisions = [1.0]
    for threshold in data:
        recalls.append(recall(data[threshold]))
        precisions.append(precision(data[threshold]))
    recalls.append(0.0)
    precisions.append(1.0)
    return recalls, precisions

def plot_metrics(data:dict, metrics=None, clients=None, reduction='mean', datasets=None, formats=['png', 'csv'], dir='', threshold=50):
    
    if clients is None:
        clients = [client for client in data]
    
    if datasets is None:
        datasets = [dataset for dataset in data[clients[0]][0]]
    
    metrics_dict = {metric: {dataset: {client: {'y': [], 'x': []} for client in clients} for dataset in datasets} for metric in metrics}
    for client in data:
        if client not in clients:
            continue
        for round in data[client]:
            for dataset in data[client][round]:
                if dataset not in datasets:
                    continue
                for metric in metrics:
                    if metric == 'loss':
                        y = data[client][round][dataset]['loss']
                        if y == None:
                            y = np.nan
                    elif metric == 'accuracy':
                        y = accuracy(data[client][round][dataset]['tpfptnfn'][threshold])
                    elif metric == 'balanced_accuracy':
                        y = balanced_accuracy(data[client][round][dataset]['tpfptnfn'][threshold])
                    elif metric == 'recall':
                        y = recall(data[client][round][dataset]['tpfptnfn'][threshold])
                    elif metric == 'precision':
                        y = precision(data[client][round][dataset]['tpfptnfn'][threshold])
                    elif metric == 'f1':
                        y = f1(data[client][round][dataset]['tpfptnfn'][threshold])
                    metrics_dict[metric][dataset][client]['y'].append(y)
                    metrics_dict[metric][dataset][client]['x'].append(round)
    
    if 'roc_curve' in metrics:
        for client in data:
            if client not in clients:
                continue
            round = max(data[client].keys())
            for dataset in data[client][round]:
                if dataset not in datasets:
                    continue
                tprs, fprs = roc_curve(data[client][round][dataset]['tpfptnfn'])
                metrics_dict['roc_curve'][dataset][client]['y'] = tprs
                metrics_dict['roc_curve'][dataset][client]['x'] = fprs
    
    if 'precision_recall_curve' in metrics:
        for client in data:
            if client not in clients:
                continue
            round = max(data[client].keys())
            for dataset in data[client][round]:
                if dataset not in datasets:
                    continue
                recalls, precisions = precision_recall_curve(data[client][round][dataset]['tpfptnfn'])
                metrics_dict['precision_recall_curve'][dataset][client]['y'] = recalls
                metrics_dict['precision_recall_curve'][dataset][client]['x'] = precisions
    
    if reduction == 'mean':
        for metric in metrics:
            for dataset in datasets:
                ys = []
                xs = []
                for client in clients:
                    y = metrics_dict[metric][dataset][client]['y']
                    ys.append(metrics_dict[metric][dataset][client]['y'])
                    xs.append(metrics_dict[metric][dataset][client]['x'])
                metrics_dict[metric][dataset]['y'] = np.mean(ys, axis=0).tolist()
                metrics_dict[metric][dataset]['y_std'] = np.std(ys, axis=0).tolist()
                metrics_dict[metric][dataset]['x'] = np.mean(xs, axis=0).tolist()
                metrics_dict[metric][dataset]['x_std'] = np.std(xs, axis=0).tolist()
                for client in clients:
                    del metrics_dict[metric][dataset][client]
    
    os.makedirs(dir, exist_ok=True)
    
    if 'png' in formats:
        for metric in metrics:
            fig = plt.figure()
            if reduction == 'mean':
                for dataset in datasets:
                    if metric == 'roc_curve' or metric == 'precision_recall_curve':
                        metrics_dict[metric][dataset]['x'], metrics_dict[metric][dataset]['y'] = zip(*sorted(zip(metrics_dict[metric][dataset]['x'], metrics_dict[metric][dataset]['y'])))
                    plt.plot(metrics_dict[metric][dataset]['x'], metrics_dict[metric][dataset]['y'], '-o', label=dataset)
                    if metric != 'roc_curve' and metric != 'precision_recall_curve':
                        plt.fill_between(metrics_dict[metric][dataset]['x'], np.array(metrics_dict[metric][dataset]['y'])-np.array(metrics_dict[metric][dataset]['y_std']), np.array(metrics_dict[metric][dataset]['y'])+np.array(metrics_dict[metric][dataset]['y_std']), alpha=0.3)
            else:
                for dataset in datasets:
                    for client in clients:
                        plt.plot(metrics_dict[metric][dataset][client]['x'], metrics_dict[metric][dataset][client]['y'], label=f'{client}: {dataset}')
            if metric == 'roc_curve':
                plt.plot([0, 1], [0, 1], linestyle='--', color='k')
                plt.xlabel('fpr')
                plt.ylabel('tpr')
                plt.title('ROC curve')
            elif metric == 'precision_recall_curve':
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.title('Precision-Recall curve')
            else:
                plt.xlabel('round')
                plt.ylabel(metric)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(dir, f'{metric}.png'))
        
    if 'csv':
        for metric in metrics:
            for dataset in datasets:
                if reduction == 'mean':
                    with open(os.path.join(dir, f'{metric}_{dataset}.csv'), 'w') as f:
                        f.write('round,mean,std\n')
                        for x, y, y_std in zip(metrics_dict[metric][dataset]['x'], metrics_dict[metric][dataset]['y'], metrics_dict[metric][dataset]['y_std']):
                            f.write(f'{x},{y},{y_std}\n')
                else:
                    for client in clients:
                        with open(os.path.join(dir, f'{metric}_{dataset}_{client}.csv'), 'w') as f:
                            f.write('round,value\n')
                            for x, y in zip(metrics_dict[metric][dataset][client]['x'], metrics_dict[metric][dataset][client]['y']):
                                f.write(f'{x},{y}\n')
                
            


