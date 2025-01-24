import yaml
import os
import pandas as pd


def main():
    EXPERIMENT = '3_banks_homo_mid'
    config_files = [
        f'experiments/{EXPERIMENT}/clients/c0/config.yaml',
        f'experiments/{EXPERIMENT}/clients/c1/config.yaml',
        f'experiments/{EXPERIMENT}/clients/c2/config.yaml',
    ]
    edges_test_dfs = []
    edges_train_dfs = []
    nodes_test_dfs = []
    nodes_train_dfs = []
    for config_file in config_files:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        for data_file, path in config['data'].items():
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(config_file), path)
            if data_file == 'edges_test':
                edges_test_dfs.append(pd.read_csv(path))
            if data_file == 'edges_train':
                edges_train_dfs.append(pd.read_csv(path))
            if data_file == 'nodes_test':
                nodes_test_dfs.append(pd.read_csv(path))
            if data_file == 'nodes_train':
                nodes_train_dfs.append(pd.read_csv(path))
    edges_test_df = pd.concat(edges_test_dfs)
    edges_train_df = pd.concat(edges_train_dfs)
    nodes_test_df = pd.concat(nodes_test_dfs)
    nodes_train_df = pd.concat(nodes_train_dfs)
    dir_path = f'./experiments/{EXPERIMENT}/data/concat_iso_data/'
    os.makedirs(dir_path, exist_ok=True)
    edges_test_df.to_csv(os.path.join(dir_path, 'edges_test.csv'))
    edges_train_df.to_csv(os.path.join(dir_path, 'edges_train.csv'))
    nodes_test_df.to_csv(os.path.join(dir_path, 'nodes_test.csv'))
    nodes_train_df.to_csv(os.path.join(dir_path, 'nodes_train.csv'))


if __name__ == '__main__':
    main()