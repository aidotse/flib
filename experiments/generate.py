import argparse
import os
import yaml
from flib.sim import DataGenerator
import pandas as pd

def main(conf_file, clients_dir=None):
    generator = DataGenerator(conf_file)
    tx_log_file = f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/data/raw/tx_log.csv' #generator()
    print(f'\nSynthetic AML data generated\n    Raw transaction log file: {tx_log_file}')
    
    if clients_dir:
        df = pd.read_csv(tx_log_file)
        banks = pd.concat([df['bankOrig'], df['bankDest']]).unique().tolist()
        banks.remove('sink')
        banks.remove('source')
        print(f'\nSplitting raw data into {len(banks)} clients: {banks}')
        
        for bank in banks:
            bank_dir = os.path.join(clients_dir, bank)
            df_bank = df[(df['bankOrig']==bank) | (df['bankDest']==bank)]
            
            # Look for config files in the client directory
            conf_file_path = os.path.join(bank_dir, 'config.yaml')
            print(f'    Looking for config files in {bank_dir}')
            if os.path.exists(conf_file_path):
                with open(conf_file_path, 'r') as f:
                    bank_conf = yaml.safe_load(f)
                print(f'    Found config file for {bank}: {conf_file_path}')
                raw_file_path = os.path.join(bank_dir, bank_conf['data']['raw'])
            else:
                print(f'    No config file found for {bank} in {bank_dir}')
                raw_file_path = os.path.join(bank_dir, 'data', 'raw', 'tx_log.csv')
            
            os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)
            df_bank.to_csv(raw_file_path, index=False)
            print(f'    Raw data for {bank} saved to {raw_file_path}')
        print()

if __name__ == "__main__":
    EXPERIMENT = '3_banks_homo_easy' #'30K_accts', '3_banks_homo_mid', '10K_accts'
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, help='Path to the config file', default=f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/data/param_files/conf.json')
    parser.add_argument('--clients_dir', type=str, help='Path to directory for client. Optional, if provided will devide the raw data over the clients.', default=f'/home/edvin/Desktop/flib/experiments/experiments/{EXPERIMENT}/clients')
    args = parser.parse_args()
    if not os.path.isabs(args.conf_file):
        args.conf_file = os.path.abspath(args.conf_file)
    if args.clients_dir:
        if not os.path.isabs(args.clients_dir):
            args.clients_dir = os.path.abspath(args.clients_dir)
    main(args.conf_file, args.clients_dir)