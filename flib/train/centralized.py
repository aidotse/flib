import pandas as pd
from flib.utils import set_random_seed
from flib.train import Clients

def centralized(train_dfs, val_dfs=[], test_dfs=[], seed=42, client='LogRegClient', **kwargs):
    
    set_random_seed(seed)
    
    # concatinat dataframes
    train_df = pd.concat(train_dfs)
    val_dfs = [val_df for val_df in val_dfs if val_df is not None]
    if val_dfs:
        val_df = pd.concat(val_dfs)
    else:
        val_df = None
    test_dfs = [test_df for test_df in test_dfs if test_df is not None]
    if test_dfs:
        test_df = pd.concat(test_dfs)
    else:
        test_df = None
    
    Client = getattr(Clients, client)
    
    client = Client(
        name=f'c0',
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        **kwargs
    )
    
    results = {client.name:  client.run(**kwargs)}
    return results
    
    