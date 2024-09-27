import argparse
from flib.preprocess import DataPreprocessor
import pandas as pd
import os

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, help='Path to the config file', default='/home/edvin/Desktop/flib/experiments/param_files/3_banks_homo_easy/conf.json')
    parser.add_argument('--tx_log_file', type=str, help='Path to raw data file', default='/home/edvin/Desktop/flib/experiments/data/3_banks_homo_easy/tx_log.csv')
    args = parser.parse_args()
    
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
        trainset, testset = dataset
        bank = trainset.loc[0, 'bank']
        sar_account_ratio_train = len(trainset[trainset['is_sar']==1]) / len(trainset[trainset['is_sar']==0])
        sar_account_ratio_test = len(testset[testset['is_sar']==1]) / len(testset[testset['is_sar']==0])
        print(f'\nsar account ratio at bank {bank}:\n    trainset: {sar_account_ratio_train}\n    testset:  {sar_account_ratio_test}')
        # save datasets
        path_to_parent_dir = os.path.abspath(os.path.dirname(args.tx_log_file))
        trainset.to_csv(os.path.join(path_to_parent_dir, 'preprocessed', f'{bank}_nodes_train.csv'), index=False)
        testset.to_csv(os.path.join(path_to_parent_dir, 'preprocessed', f'{bank}_nodes_test.csv'), index=False)
    print()
    

if __name__ == "__main__":
    main()