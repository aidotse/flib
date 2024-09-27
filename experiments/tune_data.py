import argparse

from flib.sim import DataGenerator
from flib.preprocess import DataPreprocessor
from flib.tune import DataTuner


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, help='Path to the config file', default='/home/edvin/Desktop/flib/experiments/param_files/10K_accts/conf_copy.json')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--bank', type=str, default='bank')
    args = parser.parse_args()
    
    # Create generator, preprocessor, and tuner
    generator = DataGenerator(args.conf_file)
    preprocessor = DataPreprocessor(args.conf_file, args.bank)
    tuner = DataTuner(conf_file=args.conf_file, generator=generator, preprocessor=preprocessor, operating_recall=0.50, fpr_target=0.95)
    
    # set refrence fpr
    tuner.set_fpr_max(1.0)
    
    # Tune the temporal sar parameters
    tuner(args.num_trials)


if __name__ == '__main__':
    main()