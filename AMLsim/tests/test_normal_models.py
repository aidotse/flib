
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
        self.txg.build_normal_models() # Build normal models from the base transaction types
            

def test_small_graph():
    config_str = "parameters/small_test/conf.json"
    txg = Transaction_Graph(config_str).txg
    
    # Ensure correct number of models are created
    # for forward, we try to create 1000 models but the graph only supports 60
    # each n1 -> n2 -> x has 3 different combinations
    # each n1 -> x -> n3 has 4 different combinations
    # hence, each n1 -> x -> y has 12 combinations
    # each 5 nodes have 12 possible forward patterns - 60 in total
    EXPECTED_NO_OF_MODELS = [2, 60, 2, 2, 2, 2]
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
    txg = Transaction_Graph(config_str).txg
    
    # Pick out normal models
    normal_models = dict()
    for type in TYPES:
        normal_models[type] = [nm for nm in txg.normal_models if nm.type == type]
    
    # Check Fan patterns
    NUM_DEFINED_MODELS = 10000
    max_fan_threshold = {"fan_in" : 10-1, "fan_out" : 10-1}
    min_fan_threshold = {"fan_in" : 6-1, "fan_out" : 7-1}    
    
    for type in ["fan_in", "fan_out"]:
        # find number of nodes with more than min_in_deg and min_out_deg in graph
        if type == "fan_in":
            num_candidates = len([n for n in txg.g.nodes() if txg.g.in_degree(n) >= min_fan_threshold[type]])
        else:
            num_candidates = len([n for n in txg.g.nodes() if txg.g.out_degree(n) >= min_fan_threshold[type]])
        # check how many nodes are main in multiple patterns
        main_ids = [nm.main_id for nm in normal_models[type]]
        counter = {i:main_ids.count(i) - 1 for i in main_ids}
        num_of_replicas = sum([v for _,v in counter.items()])
        
        # make sure that all candidates are used
        assert num_candidates < NUM_DEFINED_MODELS
        assert len(normal_models[type]) == (num_candidates + num_of_replicas)
        
        max_nodes_fan_in = max([len(nm.node_ids) for nm in normal_models[type]])
        min_nodes_fan_in = min([len(nm.node_ids) for nm in normal_models[type]])

        assert max_nodes_fan_in == (max_fan_threshold[type]+1)
        assert min_nodes_fan_in == (min_fan_threshold[type]+1)
    
    # Check forward patterns
    TOTAL_FORWARD_PATTERNS = 10000
    NUM_NODES_IN_FORWARD = 3
    assert len(normal_models["forward"]) == TOTAL_FORWARD_PATTERNS
    for nm in normal_models["forward"]:
        assert len(nm.node_ids) == NUM_NODES_IN_FORWARD
        
    # Check single, mutual, and periodical patterns
    TOTAL_PATTERNS = 10000
    NUM_NODES_IN_PATTERN = 2
    for type in ["single", "mutual", "periodical"]:
        assert len(normal_models[type]) == TOTAL_PATTERNS
        for nm in normal_models[type]:
            assert len(nm.node_ids) == NUM_NODES_IN_PATTERN


# define main 
if __name__ == "__main__":
    test_small_graph()
    test_large_graph()
    