import argparse
import os
from flib.sim import DataGenerator

def main(conf_file):
    generator = DataGenerator(conf_file)
    tx_log_file = generator()
    print(f'\nSynthetic AML data generated\n    Raw transaction log file: {tx_log_file}')
    pass

if __name__ == "__main__":
    DATASET = '3_banks_homo_mid' #'30K_accts', '3_banks_homo_mid'
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, help='Path to the config file', default=f'/home/edvin/Desktop/flib/experiments/param_files/{DATASET}/conf.json')
    args = parser.parse_args()
    if not os.path.isabs(args.conf_file):
        args.conf_file = os.path.abspath(args.conf_file)
    main(args.conf_file)