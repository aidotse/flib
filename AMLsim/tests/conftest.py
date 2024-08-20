import os
import json
import pytest

from scripts.transaction_graph_generator import TransactionGenerator


class Transaction_Graph:
    def __init__(self, config_str, clean = False):
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
        if not clean:
            self.txg.set_main_acct_candidates() # Identify accounts with large amounts of in and out edges
            self.txg.load_alert_patterns()  # Load alert patterns CSV file and create AML typology subgraphs
            self.txg.mark_active_edges() # mark all edges in the normal models as active
        
@pytest.fixture
def small_clean_graph():
    # Setup: Create and return a dictionary
    config_str = "parameters/small_test/conf.json"
    return Transaction_Graph(config_str, clean = True).txg

@pytest.fixture
def large_clean_graph():
    # Setup: Create and return a dictionary
    config_str = "parameters/large_test/conf.json"
    return Transaction_Graph(config_str, clean=True).txg

@pytest.fixture
def small_graph():
    # Setup: Create and return a dictionary
    config_str = "parameters/small_test/conf.json"
    return Transaction_Graph(config_str).txg

# Define a fixture
@pytest.fixture
def large_graph():
    # Setup: Create and return a dictionary
    config_str = "parameters/large_test/conf.json"
    return Transaction_Graph(config_str).txg