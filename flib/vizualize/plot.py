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
    if tp+fp == 0.0:
        return 0.0
    return tp/(tp+fn)

def precision(data):
    tp, fp, tn, fn = data['tp'], data['fp'], data['tn'], data['fn']
    if tp+fp == 0.0:
        return 1.0
    return tp/(tp+fp)

def f1(data):
    tp, fp, tn, fn = data['tp'], data['fp'], data['tn'], data['fn']
    return 2*tp/(2*tp+fp+fn)
    
def roc_curve(data):
    tprs = [] #[0]
    fprs = [] #[0]
    for threshold in data:
        tp, fp, tn, fn = data[threshold]['tp'], data[threshold]['fp'], data[threshold]['tn'], data[threshold]['fn']
        tprs.append(tp/(tp+fn))
        fprs.append(fp/(fp+tn))
    #tprs.append(1.0)
    #fprs.append(1.0)
    return tprs, fprs

def precision_recall_curve(data):
    recalls = [0.0]
    precisions = [1.0]
    for threshold in data:
        recalls.append(recall(data[threshold]))
        precisions.append(precision(data[threshold]))
    # sort by recall
    recalls, precisions = zip(*sorted(zip(recalls, precisions)))
    return recalls, precisions


def average_precision(data, recall_span=(0.0, 1.0)):
    avg_precision = 0.0
    n = 0
    for threshold in data:
        rec = recall(data[threshold])
        if rec >= recall_span[0] and rec <= recall_span[1]:
            pre = precision(data[threshold])
            avg_precision += pre
            n += 1
    avg_precision = avg_precision / n if not n == 0 else 0.0
    return avg_precision

def plot_metrics(data:dict, metrics=None, clients=None, reduction='mean', datasets=None, formats=['png', 'csv'], dir='', threshold=50, recall_span=(0.6, 1.0)):
    
    if clients is None:
        clients = [client for client in data]
    
    if datasets is None:
        datasets = [dataset for dataset in data[clients[0]][0]]
    
    metrics_dict = {metric: {dataset: {client: {'y': [], 'x': []} for client in clients} for dataset in datasets} for metric in metrics}
    
    if reduction == 'mean':
        reduced_data = {}
        for client in data:
            if client not in clients:
                continue
            for round in data[client]:
                reduced_data[round] = {}
                for dataset in data[client][round]:
                    if dataset not in datasets:
                        continue
                    loss = data[client][round][dataset]['loss']
                    loss = 0 if loss is None else loss
                    tpfptnfn = data[client][round][dataset]['tpfptnfn']
                    if not dataset in reduced_data[round]:
                        reduced_data[round][dataset] = {'loss': loss / len(clients), 'tpfptnfn': tpfptnfn}
                    else:
                        reduced_data[round][dataset]['loss'] += loss / len(clients)
                        for threshold in range(101):
                            reduced_data[round][dataset]['tpfptnfn'][threshold]['tp'] += tpfptnfn[threshold]['tp']
                            reduced_data[round][dataset]['tpfptnfn'][threshold]['fp'] += tpfptnfn[threshold]['fp']
                            reduced_data[round][dataset]['tpfptnfn'][threshold]['tn'] += tpfptnfn[threshold]['tn']
                            reduced_data[round][dataset]['tpfptnfn'][threshold]['fn'] += tpfptnfn[threshold]['fn']
        metrics_dict = {metric: {dataset: {'x': [], 'y': []} for dataset in datasets} for metric in metrics}
        for round in reduced_data:
            for dataset in reduced_data[round]:
                for metric in metrics:
                    if metric == 'loss':
                        y = reduced_data[round][dataset]['loss']
                        if y == None:
                            y = np.nan
                    elif metric == 'accuracy':
                        y = accuracy(reduced_data[round][dataset]['tpfptnfn'][threshold])
                    elif metric == 'balanced_accuracy':
                        y = balanced_accuracy(reduced_data[round][dataset]['tpfptnfn'][threshold])
                    elif metric == 'recall':
                        y = recall(reduced_data[round][dataset]['tpfptnfn'][threshold])
                    elif metric == 'precision':
                        y = precision(reduced_data[round][dataset]['tpfptnfn'][threshold])
                    elif metric == 'average_precision':
                        y = average_precision(reduced_data[round][dataset]['tpfptnfn'], recall_span)
                    elif metric == 'f1':
                        y = f1(reduced_data[round][dataset]['tpfptnfn'][threshold])
                    metrics_dict[metric][dataset]['y'].append(y)
                    metrics_dict[metric][dataset]['x'].append(round)
        if 'roc_curve' in metrics:
            round = max(reduced_data.keys())
            for dataset in reduced_data[round]:
                tprs, fprs = roc_curve(reduced_data[round][dataset]['tpfptnfn'])
                metrics_dict['roc_curve'][dataset]['y'] = tprs
                metrics_dict['roc_curve'][dataset]['x'] = fprs
        if 'precision_recall_curve' in metrics:
            round = max(reduced_data.keys())
            for dataset in reduced_data[round]:
                recalls, precisions = precision_recall_curve(reduced_data[round][dataset]['tpfptnfn'])
                metrics_dict['precision_recall_curve'][dataset]['y'] = precisions
                metrics_dict['precision_recall_curve'][dataset]['x'] = recalls
    else:
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
                        elif metric == 'average_precision':
                            y = average_precision(data[client][round][dataset]['tpfptnfn'], recall_span)
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
                    metrics_dict['precision_recall_curve'][dataset][client]['y'] = precisions
                    metrics_dict['precision_recall_curve'][dataset][client]['x'] = recalls
                    
    #if reduction == 'mean':
    #    for metric in metrics:
    #        for dataset in datasets:
    #            ys = []
    #            xs = []
    #            for client in clients:
    #                y = metrics_dict[metric][dataset][client]['y']
    #                ys.append(metrics_dict[metric][dataset][client]['y'])
    #                xs.append(metrics_dict[metric][dataset][client]['x'])
    #            for i, (y, x) in enumerate(zip(ys, xs)):
    #                if len(y) < len(max(ys, key=len)):
    #                    y.extend(y[-1:]*(len(max(ys, key=len))-len(y)))
    #                if len(x) < len(max(xs, key=len)):
    #                    x_max = max(xs, key=len)
    #                    xs[i] = x_max
    #            metrics_dict[metric][dataset]['y'] = np.mean(ys, axis=0).tolist()
    #            metrics_dict[metric][dataset]['y_std'] = np.std(ys, axis=0).tolist()
    #            metrics_dict[metric][dataset]['x'] = np.mean(xs, axis=0).tolist()
    #            metrics_dict[metric][dataset]['x_std'] = np.std(xs, axis=0).tolist()
    #            for client in clients:
    #                del metrics_dict[metric][dataset][client]
    
    os.makedirs(dir, exist_ok=True)
    os.makedirs(os.path.join(dir, 'png'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'csv'), exist_ok=True)
    
    if 'png' in formats:
        for metric in metrics:
            fig = plt.figure()
            if reduction == 'mean':
                for dataset in datasets:
                    if metric == 'roc_curve':
                        metrics_dict[metric][dataset]['x'], metrics_dict[metric][dataset]['y'] = zip(*sorted(zip(metrics_dict[metric][dataset]['x'], metrics_dict[metric][dataset]['y'])))
                    if metric == 'precision_recall_curve':
                        metrics_dict[metric][dataset]['x'], metrics_dict[metric][dataset]['y'] = zip(*sorted(zip(metrics_dict[metric][dataset]['x'], metrics_dict[metric][dataset]['y'])))
                    plt.plot(metrics_dict[metric][dataset]['x'], metrics_dict[metric][dataset]['y'], '-o', label=dataset)
                    #if metric != 'roc_curve' and metric != 'precision_recall_curve':
                    #    plt.fill_between(metrics_dict[metric][dataset]['x'], np.array(metrics_dict[metric][dataset]['y'])-np.array(metrics_dict[metric][dataset]['y_std']), np.array(metrics_dict[metric][dataset]['y'])+np.array(metrics_dict[metric][dataset]['y_std']), alpha=0.3)
            else:
                for dataset in datasets:
                    for client in clients:
                        plt.plot(metrics_dict[metric][dataset][client]['x'], metrics_dict[metric][dataset][client]['y'], '-o', label=f'{client}: {dataset}')
            if metric == 'roc_curve':
                plt.plot([0, 1], [0, 1], linestyle='--', color='k')
                plt.xlabel('fpr')
                plt.ylabel('tpr')
                plt.title('ROC curve')
            elif metric == 'precision_recall_curve':
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.title('Precision-Recall curve')
                plt.ylim(0, 0.075)
            else:
                plt.xlabel('round')
                plt.ylabel(metric)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(dir, 'png', f'{metric}.png'))
            plt.close()
        
    if 'csv':
        for metric in metrics:
            for dataset in datasets:
                if reduction == 'mean':
                    with open(os.path.join(dir, 'csv', f'{metric}_{dataset}.csv'), 'w') as f:
                        if metric == 'roc_curve':
                            f.write('fpr,tpr\n')
                        elif metric == 'precision_recall_curve':
                            f.write('recall,precision\n')
                        else:
                            f.write(f'round,{metric}\n')
                        for x, y in zip(metrics_dict[metric][dataset]['x'], metrics_dict[metric][dataset]['y']):
                            f.write(f'{x},{y}\n')
                else:
                    for client in clients:
                        with open(os.path.join(dir, 'csv', f'{metric}_{dataset}_{client}.csv'), 'w') as f:
                            if metric == 'roc_curve':
                                f.write('fpr,tpr\n')
                            elif metric == 'precision_recall_curve':
                                f.write('recall,precision\n')
                            else:
                                f.write(f'round,{metric}\n')
                            for x, y in zip(metrics_dict[metric][dataset][client]['x'], metrics_dict[metric][dataset][client]['y']):
                                f.write(f'{x},{y}\n')
                
            


