import sys
import json
import os


def insert_normal_patterns(conf_file:str):
    pass


def insert_alert_patterns(conf_file:str):
    
    with open(conf_file, 'r') as rf:
        conf = json.load(rf)
        directory = conf["input"]["directory"]
        sim_name = directory.split('/')[-1]
        patterns_to_insert = conf["input"]["insert_patterns"]
    
    src_path = os.path.join(directory, patterns_to_insert)
    dst_path = 'tmp/' + sim_name + '/alert_members.csv'
    
    with open(src_path, 'r') as src_file, open(dst_path, 'a') as dst_file:
        next(src_file)
        accounts = []
        for line in src_file:
            accounts.append(line.split(',')[2])
            dst_file.write(line)
    
    modify_accounts_file(conf_file, accounts)


def modify_accounts_file(conf_file:str, accounts:list):
    
    with open(conf_file, 'r') as rf:
        conf = json.load(rf)
        directory = conf["input"]["directory"]
        sim_name = directory.split('/')[-1]
    
    accounts_file = 'tmp/' + sim_name + '/accounts.csv'
    accounts = sorted(accounts)
    lines = []
    # find the accounts in the accounts file and write is_sar
    with open(accounts_file, 'r') as rf:
        i = 0
        for line in rf:
            acct = line.split(',')[0]
            if acct==accounts[i]:
                # replace false with true
                lines.append(line.replace('false', 'true'))
                i+=1
            else:
                lines.append(line)
    with open(accounts_file, 'w') as wf:
        for line in lines:
            wf.write(line)
    

if __name__ == "__main__":
    
    argv = sys.argv
    
    # debug 
    PARAM_FILES = '100K_accts_inserted_alerts'
    argv.append(f'paramFiles/{PARAM_FILES}/conf.json')
    
    if len(argv) == 2:
        conf_file = argv[1]
        insert_alert_patterns(conf_file)
    elif len(argv) == 3:
        conf_file = argv[1]
        type = argv[2]
        if type == 'normal':
            insert_normal_patterns(conf_file)
        elif type == 'alert':
            insert_alert_patterns(conf_file)
    else:
        print("Usage: python insert_patterns.py <config_file> [<type>]")
        print("config_file: path to the configuration file")
        print("type (optional): type of pattern to insert: 'normal' or 'alert'. Default is 'alert'")
        sys.exit(1)
    