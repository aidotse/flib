import argparse
import os
from flib.sim import DataGenerator

def main(conf_file: str):
    generator = DataGenerator(conf_file)
    tx_log_file = generator()
    print(f'\nSynthetic AML data generated\n    Raw transaction log file: {tx_log_file}')

if __name__ == "__main__":
    EXPERIMENT = '3_banks_homo_easy' 
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, help='Path to the config file', default=f'experiments/{EXPERIMENT}/data/param_files/conf.json')
    args = parser.parse_args()
    if not os.path.isabs(args.conf_file):
        args.conf_file = os.path.abspath(args.conf_file)
    main(args.conf_file)