import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np


def edge_label_hist(df:pd.DataFrame, banks:list, file:str):
    fig, ax = plt.subplots()
    width = 0.30
    x = np.array([-1*width/2, width/2])
    csv = {'bank': [], 'neg_count': [], 'pos_count': [], 'neg_ratio': [], 'pos_ratio': []}
    for i, bank in enumerate(banks):
        df_bank = df[(df['bankOrig'] == bank) & (df['bankDest'] == bank)]
        n_pos = df_bank[df_bank['isSAR'] == 1].shape[0]
        n_neg = df_bank[df_bank['isSAR'] == 0].shape[0]
        rects = ax.bar(x+i, np.array([n_neg, n_pos]), width, color=['C0', 'C1'], label=['neg', 'pos'], zorder=3)
        ax.bar_label(rects, [f'{n_neg/(n_neg+n_pos):.4f}', f'{n_pos/(n_neg+n_pos):.4f}'], padding=1)
        csv['bank'].append(bank)
        csv['neg_count'].append(n_neg)
        csv['pos_count'].append(n_pos)
        csv['neg_ratio'].append(n_neg/(n_neg+n_pos))
        csv['pos_ratio'].append(n_pos/(n_neg+n_pos))
    ax.grid(axis='y', zorder=0)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(banks)), banks)
    ax.legend(rects, ['neg', 'pos'], ncols=2)
    plt.tight_layout(pad=2.0)
    plt.title('Edge label distribution')
    plt.savefig(file)
    plt.close()
    df = pd.DataFrame(csv)
    df.to_csv(file.replace('.png', '.csv'), index=False)    
    


def node_label_hist(df:pd.DataFrame, banks:list, file:str):
    fig, ax = plt.subplots()
    width = 0.30
    x = np.array([-1*width/2, width/2])
    csv = {'bank': [], 'neg_count': [], 'pos_count': [], 'neg_ratio': [], 'pos_ratio': []}
    for i, bank in enumerate(banks):
        df_bank = df[(df['bankOrig'] == bank) & (df['bankDest'] == bank)]
        df_bank = pd.concat([df_bank[['nameOrig', 'isSAR']].rename(columns={'nameOrig': 'name'}), df_bank[['nameDest', 'isSAR']].rename(columns={'nameDest': 'name'})])
        gb = df_bank.groupby('name')
        accts = gb['isSAR'].max()
        n_pos = accts[accts == 1].shape[0]
        n_neg = accts[accts == 0].shape[0]
        rects = ax.bar(x+i, np.array([n_neg, n_pos]), width, color=['C0', 'C1'], label=['neg', 'pos'], zorder=3)
        ax.bar_label(rects, [f'{n_neg/(n_neg+n_pos):.4f}', f'{n_pos/(n_neg+n_pos):.4f}'], padding=1)
        csv['bank'].append(bank)
        csv['neg_count'].append(n_neg)
        csv['pos_count'].append(n_pos)
        csv['neg_ratio'].append(n_neg/(n_neg+n_pos))
        csv['pos_ratio'].append(n_pos/(n_neg+n_pos))
    ax.grid(axis='y', zorder=0)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(banks)), banks)
    ax.legend(rects, ['neg', 'pos'], ncols=2)
    plt.tight_layout(pad=2.0)
    plt.title('Node label distribution')
    plt.savefig(file)
    plt.close()
    df = pd.DataFrame(csv)
    df.to_csv(file.replace('.png', '.csv'), index=False)


def balance_curves(df:pd.DataFrame, file:str):
    x = np.arange(df['step'].min(), df['step'].max()+1)
    accts = pd.concat([df[['nameOrig', 'isSAR']].rename(columns={'nameOrig': 'name'}), df[['nameDest', 'isSAR']].rename(columns={'nameDest': 'name'})])
    gb = accts.groupby('name')
    accts = gb['isSAR'].max()
    accts = accts.drop([-1, -2])
    neg_accts = accts[accts == 0].sample(3).index.tolist()
    pos_accts = accts[accts == 1].sample(3).index.tolist()
    fig, ax = plt.subplots()
    csv = {'step': x}
    for acct in neg_accts:
        in_txs = df[df['nameDest'] == acct][['step', 'newbalanceDest']].rename(columns={'newbalanceDest': 'balance'})
        out_txs = df[df['nameOrig'] == acct][['step', 'newbalanceOrig']].rename(columns={'newbalanceOrig': 'balance'})
        txs = pd.concat([in_txs, out_txs])
        txs = txs.sort_values('step')
        y = np.interp(x, txs['step'], txs['balance'])
        ax.step(x, y, where='pre', color='C0', label=acct)
        csv[f'{acct}'] = y
    for acct in pos_accts:
        in_txs = df[df['nameDest'] == acct][['step', 'newbalanceDest']].rename(columns={'newbalanceDest': 'balance'})
        out_txs = df[df['nameOrig'] == acct][['step', 'newbalanceOrig']].rename(columns={'newbalanceOrig': 'balance'})
        txs = pd.concat([in_txs, out_txs])
        txs = txs.sort_values('step')
        y = np.interp(x, txs['step'], txs['balance'])
        ax.step(x, y, where='pre', color='C1', label=acct)
        csv[f'{acct}(sar)'] = y
    ax.legend()
    ax.grid()
    ax.set_xlabel('step')
    ax.set_ylabel('balance')
    plt.title('Balance curves')
    plt.tight_layout(pad=2.0)
    plt.savefig(file)
    plt.close()
    df = pd.DataFrame(csv)
    df.to_csv(file.replace('.png', '.csv'), index=False)


def pattern_hist(df:pd.DataFrame, file:str):
    # OBS! This doesnt show the dist of patterns, it shows the dist of edges belonging to a certain pattern
    # TODO: Change this to show the dist of patterns. Unclear how...
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    neg_pattern_count = df[df['isSAR'] == 0]['modelType'].value_counts()
    pos_pattern_count = df[df['isSAR'] == 1]['modelType'].value_counts()
    neg_map = {0: 'single', 1: 'fan out', 2: 'fan in', 9: 'forward', 10: 'mutual', 11: 'periodical'}
    pos_map = {1: 'fan out', 2: 'fan out', 3: 'cycle', 4: 'bipartite', 5: 'stack', 6: 'random', 7: 'scatter gather', 8: 'gather scatter'}
    fig, ax = plt.subplots()
    width = 0.30
    x = 0    
    neg_names = []
    csv = {'pattern': [], 'count': []}
    for pattern, count in neg_pattern_count.items():
        neg_names.append(neg_map[pattern])
        rect = ax.bar(x, count, width, color='C0', zorder=3)
        ax.bar_label(rect, [f'{count}'], padding=1)
        csv['pattern'].append(neg_map[pattern])
        csv['count'].append(count)
        x += 1
    pos_names = []
    for pattern, count in pos_pattern_count.items():
        pos_names.append(pos_map[pattern])
        rect = ax.bar(x, count, width, color='C1', zorder=3)
        ax.bar_label(rect, [f'{count}'], padding=1)
        x += 1
        csv['pattern'].append(f'{pos_map[pattern]} (sar)')
        csv['count'].append(count)
    ax.grid(axis='y', zorder=0)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(neg_names)+len(pos_names)), neg_names+pos_names)
    neg_proxy = plt.Rectangle((0, 0), 1, 1, fc="C0")  # Blue box for 'neg'
    pos_proxy = plt.Rectangle((0, 0), 1, 1, fc="C1")  # Orange box for 'pos'
    ax.legend([neg_proxy, pos_proxy], ['neg', 'pos'])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=2.0)
    plt.title('Pattern distribution')
    plt.savefig(file)
    plt.close()
    df = pd.DataFrame(csv)
    df.to_csv(file.replace('.png', '.csv'), index=False)


def amount_hist(df:pd.DataFrame, file:str):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    df = df[df['amount'] < 30000] # TODO: remove this, here now because 3_banks_homo_mid contains some very high sar txs, which it shouldn't
    fig, ax = plt.subplots()
    hist_neg = sns.histplot(df[df['isSAR']==0], x='amount', binwidth=100, stat='proportion', kde=True, ax=ax, color='C0')
    hist_pos = sns.histplot(df[df['isSAR']==1], x='amount', binwidth=100, stat='proportion', kde=True, ax=ax, color='C1')
    ax.grid(axis='y')
    #ax.set_yscale('log')
    ax.legend(['neg', 'pos'])
    plt.tight_layout(pad=2.0)
    plt.title('Amount distribution')
    plt.savefig(file)
    plt.close()


def spending_hist(df:pd.DataFrame, file:str):
    df = df[df['bankDest'] == 'sink']
    fig, ax = plt.subplots()
    sns.histplot(df[df['isSAR']==0], x='amount', binwidth=3000, stat='proportion', kde=False, ax=ax, color='C0')
    sns.histplot(df[df['isSAR']==1], x='amount', binwidth=3000, stat='proportion', kde=False, ax=ax, color='C1')
    ax.grid(axis='y')
    ax.set_yscale('log')
    ax.legend(['neg', 'pos'])
    plt.tight_layout(pad=2.0)
    plt.title('Spending distribution')
    plt.savefig(file)
    plt.close()


def n_txs_hist(df:pd.DataFrame, file:str):
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    df_in = df[['amount', 'nameDest', 'isSAR']].rename(columns={'nameDest': 'name'})
    df_out = df[['amount', 'nameOrig', 'isSAR']].rename(columns={'nameOrig': 'name'})
    df = pd.concat([df_in, df_out])
    d = {}
    gb = df.groupby('name')
    d['n_txs'] = gb['amount'].count()
    d['isSAR'] = gb['isSAR'].max()
    df = pd.concat(d, axis=1)
    df_neg = df[df['isSAR'] == 0]
    df_pos = df[df['isSAR'] == 1]
    
    fig, ax = plt.subplots()
    sns.histplot(df_neg, x='n_txs', binwidth=1, stat='proportion', kde=True, ax=ax, color='C0', zorder=3)
    sns.histplot(df_pos, x='n_txs', binwidth=1, stat='proportion', kde=True, ax=ax, color='C1', zorder=3)
    ax.grid(axis='y', zorder=0)
    ax.set_xscale('log')
    ax.legend(['neg', 'pos'])
    plt.tight_layout(pad=2.0)
    plt.title('Number of transactions distribution')
    plt.savefig(file)
    plt.close()


def n_spending_hist(df:pd.DataFrame, file:str):
    df = df[df['bankDest'] == 'sink']
    d = {}
    gb = df.groupby('nameOrig')
    d['n_spending_txs'] = gb['amount'].count()
    d['isSAR'] = gb['isSAR'].max()
    df = pd.concat(d, axis=1)
    df_neg = df[df['isSAR'] == 0]
    df_pos = df[df['isSAR'] == 1]
    
    fig, ax = plt.subplots()
    sns.histplot(df_neg, x='n_spending_txs', binwidth=1, stat='proportion', kde=False, ax=ax, color='C0', zorder=3)
    sns.histplot(df_pos, x='n_spending_txs', binwidth=1, stat='proportion', kde=False, ax=ax, color='C1', zorder=3)
    ax.grid(axis='y', zorder=0)
    #ax.set_xscale('log')
    ax.legend(['neg', 'pos'])
    plt.tight_layout(pad=2.0)
    plt.title('Number of spending transactions distribution')
    plt.savefig(file)


def powerlaw_degree_dist(df:pd.DataFrame, file:str):
    # OBS! This dist is not equivelent to the dist of the "blueprint". Here we look at all txs, we don't aggregate the txs between two accounts into one "edge".
    # TODO: Change this to show the dist of the "blueprint". 
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    df_in = df[['amount', 'nameDest', 'isSAR']].rename(columns={'nameDest': 'name'})
    df_out = df[['amount', 'nameOrig', 'isSAR']].rename(columns={'nameOrig': 'name'})
    df = pd.concat([df_in, df_out])
    d = {}
    gb = df.groupby('name')
    d['degree'] = gb['amount'].count()
    d['isSAR'] = gb['isSAR'].max()
    df = pd.concat(d, axis=1)
    df_neg = df[df['isSAR'] == 0]['degree'].value_counts().reset_index()
    df_pos = df[df['isSAR'] == 1]['degree'].value_counts().reset_index()
    
    def func(x, scale, gamma):
        return scale * np.power(x, -gamma)
    
    plt.figure(figsize=(10, 10))
    x = np.linspace(1, 1000, 1000)
    
    counts = df_neg['count'].values
    degrees = df_neg['degree'].values
    probs = counts / counts.sum()
    log_degrees = np.log(degrees)
    log_probs = np.log(probs)
    coeffs = np.polyfit(log_degrees, log_probs, 1)
    gamma, scale = coeffs
    plt.plot(x, func(x, np.exp(scale), -gamma), label=f'pareto sampling fit\n  gamma={-gamma:.2f}\n  scale={np.exp(scale):.2f}', color='C0')
    plt.scatter(degrees, probs, label='original neg', color='C0')
    
    counts = df_pos['count'].values
    degrees = df_pos['degree'].values
    probs = counts / counts.sum()
    log_degrees = np.log(degrees)
    log_probs = np.log(probs)
    coeffs = np.polyfit(log_degrees, log_probs, 1)
    gamma, scale = coeffs
    plt.plot(x, func(x, np.exp(scale), -gamma), label=f'pareto sampling fit\n  gamma={-gamma:.2f}\n  scale={np.exp(scale):.2f}', color='C1')
    plt.scatter(degrees, probs, label='original pos', color='C1')
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('log(degree)')
    plt.ylabel('log(probability)')
    plt.legend()
    plt.grid()
    plt.title('Powerlaw degree distribution')
    plt.savefig(file)


def graph(df:pd.DataFrame, file:str):
    pass


def plot(df:pd.DataFrame, plot_dir:str):
    banks = pd.concat([df['bankOrig'], df['bankDest']]).unique().tolist()
    if 'source' in banks:
        banks.remove('source')
    if 'sink' in banks:
        banks.remove('sink')
    edge_label_hist(df, banks, os.path.join(plot_dir, 'edge_label_hist.png'))
    node_label_hist(df, banks, os.path.join(plot_dir, 'node_label_hist.png'))
    for bank in banks:
        os.makedirs(os.path.join(plot_dir, bank), exist_ok=True)
        df_bank = df[(df['bankOrig'] == bank) | (df['bankDest'] == bank)]
        balance_curves(df_bank, os.path.join(plot_dir, bank, 'balance_curves.png'))
        pattern_hist(df_bank, os.path.join(plot_dir, bank, 'pattern_hist.png'))
        amount_hist(df_bank, os.path.join(plot_dir, bank, 'amount_hist.png'))
        spending_hist(df_bank, os.path.join(plot_dir, bank, 'spending_hist.png'))
        n_txs_hist(df_bank, os.path.join(plot_dir, bank, 'n_txs_hist.png'))
        n_spending_hist(df_bank, os.path.join(plot_dir, bank, 'n_spending_hist.png'))
        powerlaw_degree_dist(df_bank, os.path.join(plot_dir, bank, 'powerlaw_degree_dist.png'))
        graph(df_bank, os.path.join(plot_dir, bank, 'graph.png'))


def main(tx_log:str, plot_dir:str):
    np.random.seed(42)
    df = pd.read_csv(tx_log)
    os.makedirs(plot_dir, exist_ok=True)
    plot(df, plot_dir)

if __name__ == '__main__':
    DATASET = '3_banks_homo_mid' # '30K_accts', '3_banks_homo_mid'
    parser = argparse.ArgumentParser()
    parser.add_argument('--tx_log', type=str, help='Path to the transaction log', default=f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/tx_log.csv')
    parser.add_argument('--plot_dir', type=str, help='Path to the directory where the plots will be saved', default=f'/home/edvin/Desktop/flib/experiments/results/{DATASET}/data')
    args = parser.parse_args()
    main(args.tx_log, args.plot_dir)