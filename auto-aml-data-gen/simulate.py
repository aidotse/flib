import os
import json
import utils


def init_params(seed:int=0) -> dict:
    # TODO: sample with seed
    params = {
        'conf': {
            "general": {
                "random_seed": 0,
                "simulation_name": "tmp",
                "total_steps": 30
            },
            "default": {
                "min_amount": 1,
                "max_amount": 150000,
                "mean_amount": 637,
                "std_amount": 500,
                "mean_amount_sar": 637,
                "std_amount_sar": 500,
                "prob_income": 0.0,
                "mean_income": 0.0,
                "std_income": 0.0,
                "prob_income_sar": 0.0,
                "mean_income_sar": 0.0,
                "std_income_sar": 0.0,
                "mean_outcome": 200.0,
                "std_outcome": 500.0,
                "mean_outcome_sar": 200.0,
                "std_outcome_sar": 500.0,
                "prob_spend_cash": 0.7,
                "n_steps_balance_history": 7,
                "mean_phone_change_frequency": 1460,
                "std_phone_change_frequency": 365,
                "mean_phone_change_frequency_sar": 1460,
                "std_phone_change_frequency_sar": 365,
                "mean_bank_change_frequency": 1460,
                "std_bank_change_frequency": 1,
                "mean_bank_change_frequency_sar": 1460,
                "std_bank_change_frequency_sar": 1,
                "margin_ratio": 0.1,
                "prob_participate_in_multiple_sars": 0.01
            },
            "input": {
                "directory": "paramFiles/tmp",
                "schema": "schema.json",
                "accounts": "accounts.csv",
                "alert_patterns": "alertPatterns.csv",
                "normal_models": "normalModels.csv",
                "degree": "degree.csv",
                "transaction_type": "transactionType.csv",
                "is_aggregated_accounts": True
            },
            "temporal": {
                "directory": "tmp",
                "transactions": "transactions.csv",
                "accounts": "accounts.csv",
                "alert_members": "alert_members.csv",
                "normal_models": "normal_models.csv"
            },
            "output": {
                "directory": "outputs",
                "transaction_log": "tx_log.csv"
            },
            "graph_generator": {
                "degree_threshold": 1
            },
            "simulator": {
                "transaction_limit": 100000,
                "transaction_interval": 7,
                "sar_interval": 7
            },
            "scale-free": {
                "gamma": 2.0,
                "loc": 1.0,
                "scale": 1.0
            }
        },
        'accounts': [
            (10000, 1000, 100000, 'SWE', 'I', 'bank'),
        ],
        'alertPatterns': [
            (40, 'fan_out', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'TRANSFER'),
            (40, 'fan_in', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'TRANSFER'),
            (40, 'cycle', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'TRANSFER'),
            (40, 'bipartite', 2, 2, 2, 100, 1000, 1, 28, 'bank', True, 'TRANSFER'),
            (40, 'stack', 2, 4, 4, 100, 1000, 1, 28, 'bank', True, 'TRANSFER'),
            (40, 'scatter_gather', 2, 5, 5, 100, 1000, 1, 28, 'bank', True, 'TRANSFER'),
            (40, 'gather_scatter', 2, 5, 5, 100, 1000, 1, 28, 'bank', True, 'TRANSFER'),
        ],
        'normalModels': [
            (10000, 'single', 2, 2, 2, 1, 28),
            (10000, 'fan_out', 2, 2, 2, 1, 28),
            (10000, 'fan_in', 2, 2, 2, 1, 28),
            (10000, 'forward', 2, 3, 3, 1, 28),
            (10000, 'periodical', 2, 2, 2, 1, 28),
            (10000, 'mutual', 2, 2, 2, 1, 28)
        ]
    }
    return params


def create_param_files(params:dict, param_files_folder:str):
    if not os.path.exists(param_files_folder):
        os.makedirs(param_files_folder)
    
    accounts_params = params['accounts']
    with open(os.path.join(param_files_folder, 'accounts.csv'), 'w') as f:
        f.write('count,min_balance,max_balance,country,business_type,bank_id\n')
        for p in accounts_params:
            f.write(','.join(map(str, p)) + '\n')
    
    alert_patterns_params = params['alertPatterns']
    with open(os.path.join(param_files_folder, 'alertPatterns.csv'), 'w') as f:
        f.write('count,type,schedule_id,min_accounts,max_accounts,min_amount,max_amount,min_period,max_period,bank_id,is_sar,source_type\n')
        for p in alert_patterns_params:
            f.write(','.join(map(str, p)) + '\n')
    
    config_params = params['conf']
    with open(os.path.join(param_files_folder, 'conf.json'), 'w') as f:
        json.dump(config_params, f, indent=2)
    
    if 'degree' in params:
        with open(os.path.join(param_files_folder, 'degree.csv'), 'w') as f:
            f.write('Count,In-degree,Out-degree\n')
            for p in params['degree']:
                f.write(','.join(map(str, p)) + '\n')
    else:
        n = sum([p[0] for p in params['accounts']])
        gamma = config_params['scale-free']['gamma']
        loc = config_params['scale-free']['loc']
        scale = config_params['scale-free']['scale']
        values, counts = utils.powerlaw_degree_distrubution(n, gamma, loc, scale)
        with open(os.path.join(param_files_folder, 'degree.csv'), 'w') as f:
            f.write('Count,In-degree,Out-degree\n')
            for v, c in zip(values, counts):
                f.write(f'{c},{int(v[0])},{int(v[1])}\n')
        
    normal_models_params = params['normalModels']
    with open(os.path.join(param_files_folder, 'normalModels.csv'), 'w') as f:
        f.write('count,type,schedule_id,min_accounts,max_accounts,min_period,max_period,bank_id\n')
        for p in normal_models_params:
            f.write(','.join(map(str, p)) + ',\n')
    
    with open(os.path.join(param_files_folder, 'transactionType.csv'), 'w') as f:
        f.write('Type,Frequency\n')
        f.write('Transfer,1\n')


def run_simulation(config_path:str):
    os.system(f'cd ../AMLsim && python3 scripts/transaction_graph_generator.py "{config_path}"')
    os.system(f'cd ../AMLsim && mvn exec:java -Dexec.mainClass=amlsim.AMLSim -Dexec.args="{config_path}"')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    tx_log_path = os.path.join(config['output']['directory'], config['general']['simulation_name'], config['output']['transaction_log'])
    return tx_log_path


# for debugging and testing
# params = init_params()
# param_file_folder = '/home/edvin/Desktop/flib/AMLsim/paramFiles/tmp'
# create_param_files(params, param_file_folder)
# run_simulation(param_file_folder)