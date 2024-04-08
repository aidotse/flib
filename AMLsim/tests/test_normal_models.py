
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
print(os.getcwd() )

import json

from transaction_graph_generator import TransactionGenerator

class Transaction_Graph:
    def __init__(self, config_str):
        self.config_str = config_str
        
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, config_str)
        print(f"Trying to open configuration file at: {config_path}")
        with open(config_path, "r") as rf:
            conf = json.load(rf)

        self.txg = TransactionGenerator(conf, "test")
        self.txg.set_num_accounts() # Read out the number of accounts to be created
        self.txg.generate_normal_transactions()  # Load a parameter CSV file for the base transaction types
        self.txg.load_account_list()  # Load account list CSV file and write accounts to nodes in network
        self.txg.load_normal_models() # Load a parameter CSV file for Normal Models
        self.normal_models = self.txg.build_normal_models() # Build normal models from the base transaction types
            


def test_small_graph():
    config_str = "parameters/small_test/conf.json"
    txg = Transaction_Graph(config_str)
    nm = txg.normal_models
    a = 1

# define main 
if __name__ == "__main__":
    test_small_graph()
    