import pandas as pd
from utils.data import latent_dirchlet_allocation
from server import Server
from time import sleep
from modules.logisticregressor.logisticregressor import LogisticRegressor
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import random 
import numpy as np

def set_random_seed(seed:int=1):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: If you want every run to be exactly the same each time
        ##       uncomment the following lines
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    
    # Spawning method 
    mp.set_start_method('spawn')
    
    # devices
    devices = [torch.device('cuda:0')]

    '''
    # Adult dataset
    # columns = [
    #   'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
    #   'marital-status', 'occupation', 'relationship', 'race', 'sex', 
    #   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    # ]
    df = pd.read_csv('data/adult/adult.csv')
    df = df.drop(columns = ['fnlwgt', 'education'])    
    df = df.replace(' ?', None)
    df = df.dropna()
    continuous_columns = ('age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week')
    discrete_columns = ('workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country')
    target_column = 'income'
    #df1 = df[df[target_column] == ' >50K']
    #df2 = df[df[target_column] == ' <=50K']
    #df2 = df2.sample(n=len(df1), random_state=42)
    #df = pd.concat([df1, df2], axis=0)
    df = pd.get_dummies(df, columns=list(discrete_columns))
    #df, _ = train_test_split(df, test_size=0.5)
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    '''

    # AML dataset
    df = pd.read_csv('data/AML/HI-Small_Trans.csv')
    columns = [
        'Timestamp', 'From Bank', 'Account', 'To Bank', 
        'Account.1', 'Amount Received', 'Receiving Currency', 'Amount Paid', 
        'Payment Currency', 'Payment Format', 'Is Laundering']
    df = df.drop(columns=['Timestamp', 'Account', 'Account.1'])
    unique_banks = df['From Bank'].unique()
    df = df[(df['From Bank'].isin(unique_banks[0:10])) & (df['To Bank'].isin(unique_banks[0:10]))]
    continuous_columns = ('Amount Received', 'Amount Paid')
    discrete_columns = ('From Bank', 'To Bank', 'Payment Format', 'Receiving Currency', 'Payment Currency')
    target_column = 'Is Laundering'
    #df1 = df[df[target_column] == 1]
    #df2 = df[df[target_column] == 0]
    #df2 = df2.sample(n=len(df1), random_state=42)
    #df = pd.concat([df1, df2], axis=0)
    df = pd.get_dummies(df, columns=list(discrete_columns))
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Hyperparameters
    n_workers = 1 #1
    n_clients = 2 #5
    n_rounds = 201
    eval_every = 10
    n_no_aggregation_rounds = n_rounds
    Model = LogisticRegressor
    learning_rate = 0.000000001
    local_epochs = 1
    seed = 42
    
    set_random_seed(seed)
    
    server = Server(
        n_clients=n_clients, 
        n_rounds=n_rounds, 
        n_workers=n_workers, 
        Model=Model, 
        df_train=df_train, 
        df_test=df_test,
        continuous_columns=continuous_columns,
        discrete_columns=(),#discrete_columns,
        target_column=target_column,
        n_no_aggregation_rounds=n_no_aggregation_rounds,
        learning_rate = learning_rate,
        local_epochs = local_epochs,
        verbose = False,
        eval_every = eval_every,
        devices = devices
    )
    
    print('\ntraning with fl started\n')
    logs = server.run()
    print('traning with fl done\n')

    rounds = [round for round in range(0, n_rounds, eval_every)]
    for client in logs.keys():
        plt.plot(rounds, logs[client]['test']['accuracy'], color='C0')
    
    
    n_no_aggregation_rounds = n_rounds
    
    set_random_seed(seed)
    
    server = Server(
        n_clients=n_clients, 
        n_rounds=n_rounds, 
        n_workers=n_workers, 
        Model=Model, 
        df_train=df_train, 
        df_test=df_test,
        continuous_columns=continuous_columns,
        discrete_columns=(),#discrete_columns,
        target_column=target_column,
        n_no_aggregation_rounds=n_no_aggregation_rounds,
        learning_rate = learning_rate,
        local_epochs = local_epochs,
        verbose = False,
        eval_every = eval_every,
        devices = devices
    )
    print('\ntraning without fl started\n')
    logs = server.run()
    print('traning without fl done\n')
    
    rounds = [round for round in range(0, n_rounds, eval_every)]
    for client in logs.keys():
        plt.plot(rounds, logs[client]['test']['accuracy'], color='C1')
    
    plt.savefig('results.png')

    return

if __name__ == '__main__':
    main()