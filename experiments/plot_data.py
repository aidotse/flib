import os
import pandas as pd
from flib.vizualize.features import plot_features


def main():
    
    data_dir = 'experiments/data/3_banks_homo_mid/preprocessed'
    results_dir = 'experiments/results/3_banks_homo_mid/data'
    
    if not os.path.isdir(data_dir):
        print('Cant find data dir')
        pass
    
    if not os.path.isdir(results_dir):
        print('Cant find results dir')
        pass
    
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)
        
    if not os.path.isabs(results_dir):
        results_dir = os.path.abspath(results_dir)
    
    csvs = os.listdir(data_dir)
    dfs_dict = {}
    for csv in csvs:
        if csv.endswith('.csv'):
            keys = csv.split('.')[0].split('_')
            df = pd.read_csv(os.path.join(data_dir, csv))
            if keys[0] not in dfs_dict:
                dfs_dict[keys[0]] = {}
            if keys[1] not in dfs_dict[keys[0]]:
                dfs_dict[keys[0]][keys[1]] = {}
            if keys[2] not in dfs_dict[keys[0]][keys[1]]:
                dfs_dict[keys[0]][keys[1]][keys[2]] = []
            dfs_dict[keys[0]][keys[1]][keys[2]] = df
    
    plot_features(dfs_dict, results_dir, combined=False)


if __name__ == '__main__':
    main()
