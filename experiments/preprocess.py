import argparse
from flib.preprocess import DataPreprocessor
import pandas as pd
import os

def main():
    DATASET = '30K_accts' # 30K_accts, 3_banks_homo_mid
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, help='Path to the config file', default=f'/home/edvin/Desktop/flib/experiments/param_files/{DATASET}/conf.json')
    parser.add_argument('--tx_log_file', type=str, help='Path to raw data file', default=f'/home/edvin/Desktop/flib/experiments/data/{DATASET}/tx_log.csv')
    args = parser.parse_args()
    
    print(args.conf_file)
    print(args.tx_log_file)
    
    if not os.path.isabs(args.conf_file):
        args.conf_file = os.path.abspath(args.conf_file)
    if not os.path.isabs(args.tx_log_file):
        args.conf_file = os.path.abspath(args.tx_log_file)
    
    # check sar tx ratio 
    df = pd.read_csv(args.tx_log_file)
    df = df[df['bankOrig'] != 'source']
    df = df[df['bankDest'] != 'sink']
    sar_tx_ratio = len(df[df['isSAR']==1]) / len(df[df['isSAR']==0])
    print(f'\nsar transaction ratio: {sar_tx_ratio}\n')
     
    preprocessor = DataPreprocessor(args.conf_file)
    datasets = preprocessor(args.tx_log_file)
    
    # check sar account ratio
    for dataset in datasets:
        df_nodes_train, df_nodes_test, df_edges_train, df_edges_test = dataset
        bank = df_nodes_train.loc[0, 'bank']
        sar_account_ratio_train = len(df_nodes_train[df_nodes_train['is_sar']==1]) / (len(df_nodes_train[df_nodes_train['is_sar']==0]) + len(df_nodes_train[df_nodes_train['is_sar']==1]))
        sar_account_ratio_test = len(df_nodes_test[df_nodes_test['is_sar']==1]) / (len(df_nodes_test[df_nodes_test['is_sar']==0]) + len(df_nodes_test[df_nodes_test['is_sar']==1]))
        print(f'\nsar account ratio at bank {bank}:\n    trainset: {sar_account_ratio_train}\n    testset:  {sar_account_ratio_test}')
        # save datasets
        path_to_parent_dir = os.path.abspath(os.path.dirname(args.tx_log_file))
        # make dir
        os.makedirs(os.path.join(path_to_parent_dir, 'preprocessed'), exist_ok=True)
        df_nodes_train.to_csv(os.path.join(path_to_parent_dir, 'preprocessed', f'{bank}_nodes_train.csv'), index=False)
        df_nodes_test.to_csv(os.path.join(path_to_parent_dir, 'preprocessed', f'{bank}_nodes_test.csv'), index=False)
        df_edges_train.to_csv(os.path.join(path_to_parent_dir, 'preprocessed', f'{bank}_edges_train.csv'), index=False)
        df_edges_test.to_csv(os.path.join(path_to_parent_dir, 'preprocessed', f'{bank}_edges_test.csv'), index=False)
    print()
    

if __name__ == "__main__":
    main()