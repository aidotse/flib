# wrapper for AMLsim

import json
import os

from flib.sim.AMLsim.scripts.generate_scalefree import generate_degree_file
from flib.sim.AMLsim.scripts.transaction_graph_generator import generate_transaction_graph

class DataGenerator:
    def __init__(self, conf_file):
        self.conf_file = conf_file

    def __call__(self, conf_file=None):
        
        if conf_file is None:
            conf_file = self.conf_file
            
        print(f'Simulating with {conf_file}')
        
        # get path to working dir
        dir = os.path.dirname(os.path.realpath(__file__))
        
        # check if degree.csv exists
        with open(conf_file, 'r') as f:
            config = json.load(f)
        degree_path = os.path.join(config['input']['directory'], config['input']['degree'])
        # create degree.csv if it does not exist
        if not os.path.exists(degree_path):
           generate_degree_file(conf_file)
        
        # generate transaction graph and run simulation
        generate_transaction_graph(conf_file)
        os.system(f'cd {dir}/AMLsim && mvn exec:java -Dexec.mainClass=amlsim.AMLSim -Dexec.args="{conf_file }"')

        with open(conf_file , 'r') as f:
            config = json.load(f)
        tx_log_path = os.path.join(config['output']['directory'], config['output']['transaction_log'])
        
        return tx_log_path