# wrapper for AMLsim

import json
import os

from flib.sim.AMLsim.scripts.generate_scalefree import generate_degree_file
from flib.sim.AMLsim.scripts.transaction_graph_generator import generate_transaction_graph

class DataGenerator:
    def __init__(self, conf_file):
        self.conf_file = conf_file

    def run(self):
        
        print(f'Simulating with {self.conf_file}')
        
        # get path to working dir
        dir = os.path.dirname(os.path.realpath(__file__))
        
        # check if degree.csv exists
        with open(self.conf_file, 'r') as f:
            config = json.load(f)
        degree_path = os.path.join(config['input']['directory'], config['input']['degree'])
        # create degree.csv if it does not exist
        if not os.path.exists(degree_path):
           generate_degree_file(self.conf_file)
        
        # generate transaction graph and run simulation
        #os.system(f'cd {dir}/AMLsim && python3 scripts/transaction_graph_generator.py "{self.conf_file }"')
        generate_transaction_graph(self.conf_file)
        os.system(f'cd {dir}/AMLsim && mvn exec:java -Dexec.mainClass=amlsim.AMLSim -Dexec.args="{self.conf_file }"')

        with open(self.conf_file , 'r') as f:
            config = json.load(f)
        tx_log_path = os.path.join(config['output']['directory'], config['general']['simulation_name'], config['output']['transaction_log'])
        
        return tx_log_path