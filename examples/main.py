import argparse

from flib.sim import DataGenerator
from flib.tune import DataTuner


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, help='Path to the config file', default='/home/edvin/Desktop/flib/examples/param_files/3_banks_homo_hard/conf.json')
    args = parser.parse_args()
    
    # Generate raw data
    dg = DataGenerator(args.conf_file)
    dg.run()
    
    


if __name__ == '__main__':
    main()