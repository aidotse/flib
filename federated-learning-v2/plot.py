import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dask.dataframe as dd
from ast import literal_eval
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import multiprocessing as mp
import seaborn as sns

def load_data(log_file):
    data = pd.read_csv(log_file, sep=';')
    return data

def calculate_accuracy(row) -> float:
    return accuracy_score(literal_eval(row['y_true']), literal_eval(row['y_pred']))

def calculate_balanced_accuracy(row) -> float:
    return balanced_accuracy_score(literal_eval(row['y_true']), literal_eval(row['y_pred']))

def calculate_precision(row) -> float:
    return precision_score(literal_eval(row['y_true']), literal_eval(row['y_pred']), zero_division=0)

def calculate_recall(row) -> float:
    return recall_score(literal_eval(row['y_true']), literal_eval(row['y_pred']))

def calculate_f1(row) -> float:
    return f1_score(literal_eval(row['y_true']), literal_eval(row['y_pred']))

def calculate_confusion_matrix(row) -> list:
    return confusion_matrix(literal_eval(row['y_true']), literal_eval(row['y_pred']))

def calculate_tnfpfntp(row):
    tn, fp, fn, tp = confusion_matrix(literal_eval(row['y_true']), literal_eval(row['y_pred'])).ravel()
    return pd.Series([tn, fp, fn, tp])
    
def calculate_metrics(df):
    df['accuracy'] = df.apply(calculate_accuracy, axis=1)
    df['balanced_accuracy'] = df.apply(calculate_balanced_accuracy, axis=1)
    df['precision'] = df.apply(calculate_precision, axis=1)
    df['recall'] = df.apply(calculate_recall, axis=1)
    df['f1'] = df.apply(calculate_f1, axis=1)
    df[['tn', 'fp', 'fn', 'tp']] = df.apply(calculate_tnfpfntp, axis=1)
    return df
    
def calculate_metrics_mp(df):
    n_workers = 8
    df_splits = np.array_split(df, n_workers)
    with mp.Pool(n_workers) as p:
        dfs = p.map(calculate_metrics, df_splits)
    df = pd.concat(dfs)
    df.drop(columns=['y_pred', 'y_true'], inplace=True)
    return df

def avrage_metrics(df):
    tmp = pd.DataFrame()
    metrics = df.columns[3:]
    for metric in metrics:
        tmp[metric+'_mean'] = df.groupby(['round', 'type'])[metric].mean()
        tmp[metric+'_std'] = df.groupby(['round', 'type'])[metric].std()
    df = tmp.reset_index()
    return df
    
def get_curves(df, type, metric, client=None):
    print(df[df['type']==type].head())

def plot(*log_files, types=['train', 'test'], metrics=['loss', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1'], labels=['fed', 'iso', 'cen']):
        
    # load the data
    print('loading data...', end=' ')
    dfs = []
    for log_file in log_files:
        df = load_data(log_file)
        df = calculate_metrics_mp(df)
        df = avrage_metrics(df)
        dfs.append(df)
    print('done')
    
    # plot metrics
    n_subplots = len(metrics)
    n_rows = int(np.sqrt(n_subplots))
    n_cols = int(np.ceil(n_subplots/n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    for ax, metric in zip(axs.flat, metrics):
        for type in types:
            for df, label in zip(dfs, labels):
                df = df[df['type']==type]
                ax.plot(df['round'], df[metric+'_mean'], label=label+'_'+type)
                ax.fill_between(df['round'], df[metric+'_mean']-df[metric+'_std'], df[metric+'_mean']+df[metric+'_std'], alpha=0.2)
        ax.set_xlabel('round', fontsize=16)
        ax.set_ylabel(metric, fontsize=16)
        ax.legend()
    plt.tight_layout()
    plt.savefig('results/metrics.png')
    
    # plot confusion matrix from test in last run
    fig, axs = plt.subplots(1, len(labels), figsize=(15, 5))
    cmaps = ['Blues', 'Oranges', 'Greens', 'Greys']
    for i, df in enumerate(dfs):
        df_tmp = df[df['type']=='test']
        df_tmp = df_tmp[df_tmp['round']==df_tmp['round'].max()]
        sns.heatmap(df_tmp[['tn_mean', 'fp_mean', 'fn_mean', 'tp_mean']].values.reshape(2, 2), annot=True, annot_kws={"size": 16}, fmt='g', ax=axs[i], cmap=cmaps[i], cbar=False)
        axs[i].set_title(labels[i], fontsize=18)
        axs[i].set_xlabel('prediction', fontsize=16)
        axs[i].set_ylabel('truth', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    
    pass

if __name__ == "__main__":
    plot('results/param_sweep_fed/run_5/log', 'results/param_sweep_iso/run_8/log', 'results/param_sweep_cen/run_0/log', types=['test'], metrics=['loss', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1'])