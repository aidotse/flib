import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import umap
from typing import List, Union
import argparse

def plot_umap(X:np.ndarray, y:np.ndarray) -> plt.Figure:
    points = umap.UMAP().fit_transform(X)
    fig, ax = plt.subplots()
    colors = np.where(y == 0, 'C0', 'C1')
    ax.scatter(x=points[:,0], y=points[:,1], c=colors)
    return fig

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
                

def plot(df:pd.DataFrame) -> List[plt.Figure]:
    figs = []
    figs.append(plot_umap(df))
    return figs


if __name__ == "__main__":
    
    EXPERIMENT = "3_banks_homo_mid"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", help="Paths to data files.", default=[
        f"experiments/{EXPERIMENT}/data/preprocessed/nodes_train.csv",
        f"experiments/{EXPERIMENT}/clients/c0/data/preprocessed/nodes_train.csv",
        f"experiments/{EXPERIMENT}/clients/c1/data/preprocessed/nodes_train.csv",
        f"experiments/{EXPERIMENT}/clients/c2/data/preprocessed/nodes_train.csv",
    ])
    parser.add_argument("--dir", type=str, help="Path to the directory where the plots will be saved.", default=f"experiments/{EXPERIMENT}/results/data/preprocessed")
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    file = args.files[0]
    df = pd.read_csv(file)
    X = df.iloc[:,2:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()
    fig = plot_umap(X, y)
    fig.savefig(os.path.join(args.dir, "umap.png"))
    
    os.makedirs(os.path.join(args.dir, "c0"), exist_ok=True)
    file = args.files[1]
    df = pd.read_csv(file)
    X = df.iloc[:,2:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()
    fig = plot_umap(X, y)
    fig.savefig(os.path.join(args.dir, "c0", "umap.png"))
    
    os.makedirs(os.path.join(args.dir, "c1"), exist_ok=True)
    file = args.files[2]
    df = pd.read_csv(file)
    X = df.iloc[:,2:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()
    fig = plot_umap(X, y)
    fig.savefig(os.path.join(args.dir, "c1", "umap.png"))
    
    os.makedirs(os.path.join(args.dir, "c2"), exist_ok=True)
    file = args.files[3]
    df = pd.read_csv(file)
    X = df.iloc[:,2:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()
    fig = plot_umap(X, y)
    fig.savefig(os.path.join(args.dir, "c2", "umap.png"))
