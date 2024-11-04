import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def plot_hist(df):
    pos = df[df['is_sar'] == 1]
    neg = df[df['is_sar'] == 0]

def plot_features(dfs_dict, results_dir, combined):
    
    for key1 in dfs_dict:
        for key2 in dfs_dict[key1]:
            for key3 in dfs_dict[key1][key2]:
                df = dfs_dict[key1][key2][key3]
                os.makedirs(os.path.join(results_dir, key1, key2, key3), exist_ok=True)
                
                #with open(os.path.join(results_dir, key1, key2, key3, 'description.txt'), 'w') as f:
                #    f.write(df.describe().to_string())
                
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='is_sar', stat='percent', ax=ax, zorder=3)
                ax.grid(axis='y', zorder=0)
                plt.ylim(0, 100)
                plt.title('Label distribution')
                plt.savefig(os.path.join(results_dir, key1, key2, key3, 'label_count.png'))
                plt.close()
                
                #columns = df.columns.tolist()
                #g = sns.PairGrid(df[columns[1:]], hue='is_sar')
                #g.map_upper(sns.histplot)
                #g.map_lower(sns.kdeplot, fill=True)
                #g.map_diag(sns.histplot, kde=True)
                #plt.savefig(os.path.join(results_dir, key1, key2, key3, 'hist.png'))
                
                
    
if __name__ == '__main__':
    plot_features()

