import argparse

from flib.sim import DataGenerator
from flib.preprocess import DataPreprocessor
from flib.tune import DataTuner
from time import time

def main():
    
    DATASET = '1_bank_homo_mid'
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, help='Path to the config file', default=f'/home/edvin/Desktop/flib/experiments/param_files/{DATASET}/conf.json')
    parser.add_argument('--num_trials', type=int, default=200)
    parser.add_argument('--utility', type=str, default='ap')
    parser.add_argument('--bank', type=str, default='bank')
    args = parser.parse_args()
    
    # Create generator, preprocessor, and tuner
    generator = DataGenerator(args.conf_file)
    preprocessor = DataPreprocessor(args.conf_file, args.bank)
    tuner = DataTuner(conf_file=args.conf_file, generator=generator, preprocessor=preprocessor, target=0.01, utility=args.utility, model='DecisionTreeClassifier')
    
    # Tune the temporal sar parameters
    t = time()
    tuner(args.num_trials)
    t = time() - t 
    print(f'\nExec time: {t}\n')

if __name__ == '__main__':
    main()