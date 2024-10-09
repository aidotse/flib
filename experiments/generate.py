import argparse
from flib.sim import DataGenerator

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, help='Path to the config file', default='/home/edvin/Desktop/flib/experiments/param_files/3_banks_homo_mid/conf.json')
    args = parser.parse_args()
    
    generator = DataGenerator(args.conf_file)
    tx_log_file = generator()
    print(f'\nSynthetic AML data generated\n    Raw transaction log file: {tx_log_file}')
    pass

if __name__ == "__main__":
    main()