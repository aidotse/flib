
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))


import json

from transaction_graph_generator import TransactionGenerator

TYPES = ["single", "forward", "mutual", "periodical", "fan_in", "fan_out"]

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
    
    # Ensure correct number of models are created
    EXPECTED_NO_OF_MODELS = [2, 2, 2, 2, 2, 2]
    normal_models = dict()
    for type, expected_num in zip(TYPES,EXPECTED_NO_OF_MODELS):
        normal_models[type] = [nm for nm in txg.normal_models if nm.type == type]
        assert len(normal_models[type]) == expected_num
    
    # Ensure fan patterns have correct number of nodes
    EXPECTED_NO_OF_NODES = [3,5]
    for nm, expected_num in zip(normal_models["fan_in"], EXPECTED_NO_OF_NODES):
        assert len(nm.node_ids) == expected_num
    for nm, expected_num in zip(normal_models["fan_out"], EXPECTED_NO_OF_NODES):
        assert len(nm.node_ids) == expected_num
    
def test_large_graph():
    config_str = "parameters/large_test/conf.json"
    txg = Transaction_Graph(config_str)
    
    EXPECTED_NO_OF_MODELS = [1e4]*6
    normal_models = dict()
    for type, expected_num in zip(TYPES,EXPECTED_NO_OF_MODELS):
        normal_models[type] = [nm for nm in txg.normal_models if nm.type == type]
        assert len(normal_models[type]) == expected_num
    
    # Check that we obtain the max and min number of nodes in fan patterns
    MAX_FAN_IN = 10
    MIN_FAN_IN = 3
    for type in ["fan_in", "fan_out"]:
        max_nodes_fan_in = max([len(nm.node_ids) for nm in normal_models[type]])
        min_nodes_fan_in = min([len(nm.node_ids) for nm in normal_models[type]])
    
        assert max_nodes_fan_in == MAX_FAN_IN
        assert min_nodes_fan_in == MIN_FAN_IN
        


# define main 
if __name__ == "__main__":
    test_small_graph()
    test_large_graph()
    