import os
import json

def create_param_files(params:dict, param_file_folder:str):
    if not os.path.exists(param_file_folder):
        os.makedirs(param_file_folder)
    config_params = params['config']
    with open(os.path.join(param_file_folder, 'config.json'), 'w') as f:
        json.dump(config_params, f, indent=2)
    config_params = params['accounts']
    


params = {
    'config': {
        'general': {
            'random_seed': 0, 
            'simulation_name': '100K_accts', 
            'total_steps': 367
        }, 
        'default': {
            'min_amount': 1, 
            'max_amount': 150000, 
            'mean_amount': 637, 
            'std_amount': 500, 
            'mean_amount_sar': 1000, 
            'std_amount_sar': 500, 
            'prob_income': 0.0, 
            'mean_income': 0.0, 
            'std_income': 0.0, 
            'prob_income_sar': 0.0, 
            'mean_income_sar': 0.0, 
            'std_income_sar': 0.0, 
            'mean_outcome': 200.0, 
            'std_outcome': 500.0, 
            'mean_outcome_sar': 200.0, 
            'std_outcome_sar': 500.0, 
            'prob_spend_cash': 0.7, 
            'n_steps_balance_history': 56, 
            'mean_phone_change_frequency': 1460, 
            'std_phone_change_frequency': 365, 
            'mean_phone_change_frequency_sar': 365, 
            'std_phone_change_frequency_sar': 182, 
            'mean_bank_change_frequency': 1460, 
            'std_bank_change_frequency': 1, 
            'mean_bank_change_frequency_sar': 1460, 
            'std_bank_change_frequency_sar': 1, 
            'margin_ratio': 0.1, 
            'prob_participate_in_multiple_sars': 0.2
        }
    },
    'accounts': [
        (27700, 1000, 100000, 'SWE', 'I', 'swedbank'),
        (13480, 1000, 100000, 'SWE', 'I', 'handelsbanken'),
        (58820, 1000, 100000, 'SWE', 'I', 'other')
    ],
    'normalModels': {
        ()
    }
}

param_file_folder = 'param_files/test'

create_param_files(params, param_file_folder)