import argparse
from flib.preprocess import DataPreprocessor
import pandas as pd
import os
import yaml
import multiprocessing as mp

def worker(config):
    preprocessor = DataPreprocessor(config['preprocess'])
    dataset_dict = preprocessor(config['data']['raw'])
    for set, data in dataset_dict.items():
        data.to_csv(config['data'][set], index=False)

def main(config_files, n_workers=1):
    configs = []
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        for file, path in config['data'].items():
            if not os.path.isabs(path):
                config['data'][file] = os.path.join(os.path.dirname(config_file), path)
            if file == 'raw':
                if not os.path.exists(config['data'][file]):
                    raise FileNotFoundError(f'Raw data file not found: {config["data"]["raw"]}')
            else:
                if not os.path.exists(config['data'][file]):
                    os.makedirs(os.path.dirname(config['data'][file]), exist_ok=True)
        configs.append(config)
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            pool.map(worker, configs)
    else:
        for config in configs:
            worker(config)

if __name__ == "__main__":
    EXPERIMENT = '3_banks_homo_easy'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+', type=str, help='Path to the config files for preprocessing.', default=[
        f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/data/config.yaml',
        #f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/clients/c0/config.yaml',
        #f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/clients/c1/config.yaml',
        #f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/clients/c2/config.yaml',
    ])
    parser.add_argument('--n_workers', type=int, help='Number of workers for preprocessing several datasets at once, default = 1.', default=4)
    args = parser.parse_args()
    
    main(args.config_files, args.n_workers)