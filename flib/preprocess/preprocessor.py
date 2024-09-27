import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os

class DataPreprocessor:
    def __init__(self, conf_file, bank=None):
        with open(conf_file) as f:
            conf = json.load(f)
        self.num_windows = conf["preprocessing"]["num_windows"]
        self.window_len = conf["preprocessing"]["window_len"]
        self.train_start_step = conf["preprocessing"]["train_start_step"]
        self.train_end_step = conf["preprocessing"]["train_end_step"]
        self.test_start_step = conf["preprocessing"]["test_start_step"]
        self.test_end_step = conf["preprocessing"]["test_end_step"]
        self.include_edges = conf["preprocessing"]["include_edges"]
        self.bank = bank
    
    
    def load_data(self, path:str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df[df['bankOrig'] != 'source'] # TODO: create features based on source transactions
        df.drop(columns=['type', 'oldbalanceOrig', 'oldbalanceDest', 'newbalanceOrig', 'newbalanceDest', 'alertID', 'modelType'], inplace=True)
        return df

    
    def cal_node_features(self, df:pd.DataFrame, start_step, end_step, bank) -> pd.DataFrame:
        if self.num_windows * self.window_len < end_step - start_step:
            raise ValueError(f'Number of windows {self.num_windows} and the windows length {self.window_len} do not allow coverage of the whole dataset. Inceasing number of windows or length of windows')
        window_overlap = (self.num_windows * self.window_len - (end_step - start_step + 1)) // (self.num_windows - 1)
        windows = [(start_step + i*(self.window_len-window_overlap), start_step + i*(self.window_len-window_overlap) + self.window_len-1) for i in range(self.num_windows)]
        windows[-1] = (end_step - self.window_len + 1, end_step)
        # filter out transactions to the sink
        df_spending = df[df['bankDest'] == 'sink'].rename(columns={'nameOrig': 'account'})
        # filter out and reform transactions within the network 
        df_network = df[df['bankDest'] != 'sink']
        df_in = df_network[['step', 'nameDest', 'bankDest', 'amount', 'nameOrig', 'daysInBankDest', 'phoneChangesDest', 'isSAR']].rename(columns={'nameDest': 'account', 'bankDest': 'bank', 'nameOrig': 'counterpart', 'daysInBankDest': 'days_in_bank', 'phoneChangesDest': 'n_phone_changes', 'isSAR': 'is_sar'})
        df_out = df_network[['step', 'nameOrig', 'bankOrig', 'amount', 'nameDest', 'daysInBankOrig', 'phoneChangesOrig', 'isSAR']].rename(columns={'nameOrig': 'account', 'bankOrig': 'bank', 'nameDest': 'counterpart', 'daysInBankOrig': 'days_in_bank', 'phoneChangesOrig': 'n_phone_changes', 'isSAR': 'is_sar'})

        df_nodes = pd.DataFrame()
        df_nodes = pd.concat([df_out[['account', 'bank']], df_in[['account', 'bank']]]).drop_duplicates().set_index('account')
        node_features = {}
        
        # calculate spending features
        for window in windows:
            gb_spending = df_spending[(df_spending['step']>=window[0])&(df_spending['step']<=window[1])].groupby(['account'])
            node_features[f'sums_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].sum()
            node_features[f'means_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].mean()
            node_features[f'medians_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].median()
            node_features[f'stds_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].std()
            node_features[f'maxs_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].max()
            node_features[f'mins_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].min()
            node_features[f'counts_spending_{window[0]}_{window[1]}'] = gb_spending['amount'].count()
            gb_in = df_in[(df_in['step']>=window[0])&(df_in['step']<=window[1])].groupby(['account'])
            node_features[f'sum_in_{window[0]}_{window[1]}'] = gb_in['amount'].apply(lambda x: x[x > 0].sum())
            node_features[f'mean_in_{window[0]}_{window[1]}'] = gb_in['amount'].mean()
            node_features[f'median_in_{window[0]}_{window[1]}'] = gb_in['amount'].median()
            node_features[f'std_in_{window[0]}_{window[1]}'] = gb_in['amount'].std()
            node_features[f'max_in_{window[0]}_{window[1]}'] = gb_in['amount'].max()
            node_features[f'min_in_{window[0]}_{window[1]}'] = gb_in['amount'].min()
            node_features[f'count_in_{window[0]}_{window[1]}'] = gb_in['amount'].count()
            node_features[f'count_unique_in_{window[0]}_{window[1]}'] = gb_in['counterpart'].nunique()
            gb_out = df_out[(df_out['step']>=window[0])&(df_out['step']<=window[1])].groupby(['account'])
            node_features[f'sum_out_{window[0]}_{window[1]}'] = gb_out['amount'].apply(lambda x: x[x > 0].sum())
            node_features[f'mean_out_{window[0]}_{window[1]}'] = gb_out['amount'].mean()
            node_features[f'median_out_{window[0]}_{window[1]}'] = gb_out['amount'].median()
            node_features[f'std_out_{window[0]}_{window[1]}'] = gb_out['amount'].std()
            node_features[f'max_out_{window[0]}_{window[1]}'] = gb_out['amount'].max()
            node_features[f'min_out_{window[0]}_{window[1]}'] = gb_out['amount'].min()
            node_features[f'count_out_{window[0]}_{window[1]}'] = gb_out['amount'].count()
            node_features[f'count_unique_out_{window[0]}_{window[1]}'] = gb_out['counterpart'].nunique()
        # calculate non window related features
        df_combined = pd.concat([df_in[['account', 'days_in_bank', 'n_phone_changes', 'is_sar']], df_out[['account', 'days_in_bank', 'n_phone_changes', 'is_sar']]])
        gb = df_combined.groupby('account')
        node_features['counts_days_in_bank'] = gb['days_in_bank'].max()
        node_features['counts_phone_changes'] = gb['n_phone_changes'].max()
        # find label
        node_features['is_sar'] = gb['is_sar'].max()
        # concat features
        node_features_df = pd.concat(node_features, axis=1)
        # merge with nodes
        df_nodes = df_nodes.join(node_features_df)
        # filter out nodes not belonging to the bank
        df_nodes = df_nodes[df_nodes['bank'] == bank] # TODO: keep these nodes? see TODO below about get edges
        # if any value is nan, there was no transaction in the window for that account and hence the feature should be 0
        df_nodes = df_nodes.fillna(0.0)
        # check if there is any missing values
        assert df_nodes.isnull().sum().sum() == 0, 'There are missing values in the node features'
        # check if there are any negative values in all comuns except the bank column
        assert (df_nodes.drop(columns='bank') < 0).sum().sum() == 0, 'There are negative values in the node features'
        return df_nodes
    
    
    def cal_edge_features(df:pd.DataFrame, directional:bool=False) -> pd.DataFrame:
        # calculate start and end step for each window
        if type(windows) == int:
            n_windows = windows
            start_step, end_step = df['step'].min(), df['step'].max()
            steps_per_window = (end_step - start_step) // n_windows
            windows = [(start_step + i*steps_per_window, start_step + (i+1)*steps_per_window-1) for i in range(n_windows)]
            windows[-1] = (windows[-1][0], end_step)
        elif type(windows) == tuple:
            num_windows, window_len = windows
            start_step, end_step = df['step'].min(), df['step'].max()
            if num_windows * window_len < end_step - start_step:
                raise ValueError(f'Number of windows {num_windows} and the windows length {window_len} do not allow coverage of the whole dataset. Inceasing number of windows or length of windows')
            window_overlap = (num_windows * window_len - (end_step - start_step + 1)) // (num_windows - 1)
            windows = [(start_step + i*(window_len-window_overlap), start_step + i*(window_len-window_overlap) + window_len-1) for i in range(num_windows)]
            windows[-1] = (end_step - window_len + 1, end_step)
        # filter out transactions to the sink
        df = df[df['bankDest'] != 'sink']
        # rename
        df = df[['step', 'nameOrig', 'nameDest', 'amount', 'isSAR']].rename(columns={'nameOrig': 'src', 'nameDest': 'dst', 'isSAR': 'is_sar'})
        # if directional=False then sort src and dst
        if not directional:
            df[['src', 'dst']] = np.sort(df[['src', 'dst']], axis=1)
        # init final dataframe
        df_edges = pd.DataFrame()
        for window in windows:
            gb = df[(df['step']>=window[0])&(df['step']<=window[1])].groupby(['src', 'dst'])
            df_edges[f'sums_{window[0]}_{window[1]}'] = gb['amount'].sum()
            df_edges[f'means_{window[0]}_{window[1]}'] = gb['amount'].mean()
            df_edges[f'medians_{window[0]}_{window[1]}'] = gb['amount'].median()
            df_edges[f'stds_{window[0]}_{window[1]}'] = gb['amount'].std().fillna(0.0)
            df_edges[f'maxs_{window[0]}_{window[1]}'] = gb['amount'].max()
            df_edges[f'mins_{window[0]}_{window[1]}'] = gb['amount'].min()
            df_edges[f'counts_{window[0]}_{window[1]}'] = gb['amount'].count()
        # find label
        gb = df.groupby(['src', 'dst'])
        df_edges[f'is_sar'] = gb['is_sar'].max()
        df_edges.reset_index(inplace=True)
        # if any value is nan, there was no transaction in the window for that edge and hence the feature should be 0
        df_edges = df_edges.fillna(0.0)
        # check if there is any missing values
        assert df_edges.isnull().sum().sum() == 0, 'There are missing values in the edge features'
        # check if there are any negative values in all comuns except the bank column
        assert (df_edges.drop(columns=['src', 'dst']) < 0).sum().sum() == 0, 'There are negative values in the edge features'
        return df_edges

    
    def preprocess(self, df:pd.DataFrame, bank):
        df_train = df[(df['step'] >= self.train_start_step) & (df['step'] <= self.train_end_step)]
        df_test = df[(df['step'] >= self.test_start_step) & (df['step'] <= self.test_end_step)]
        df_nodes_train = self.cal_node_features(df_train, self.train_start_step, self.train_end_step, bank)
        df_nodes_test = self.cal_node_features(df_test, self.test_start_step, self.test_end_step, bank)
        df_nodes_train.reset_index(inplace=True)
        df_nodes_test.reset_index(inplace=True)
        
        if self.include_edges:
            df_edges_train = self.cal_edge_features(df_train[(df_train['bankOrig']==bank) & (df_train['bankDest']==bank)], directional=True) # TODO: enable edges to/from the bank? the node features use these txs but unclear how to ceate a edge in this case, the edge can't be connected to a node with node features (could create node features based on edge txs, then the node features and edge features will look the same and some node features will be missing)
            df_edges_test = self.cal_edge_features(df_test[(df_test['bankOrig']==bank) & (df_test['bankDest']==bank)], directional=True)
            node_to_index = pd.Series(df_nodes_train.index, index=df_nodes_train['account']).to_dict()
            df_edges_train['src'] = df_edges_train['src'].map(node_to_index) # OBS: in the csv files it looks like the edge src refers to the node two rows above the acculat node, this is due to the column head and that it starts counting at 0
            df_edges_train['dst'] = df_edges_train['dst'].map(node_to_index)
            node_to_index = pd.Series(df_nodes_test.index, index=df_nodes_test['account']).to_dict()
            df_edges_test['src'] = df_edges_test['src'].map(node_to_index)
            df_edges_test['dst'] = df_edges_test['dst'].map(node_to_index)
            return (df_nodes_train, df_nodes_test, df_edges_train, df_edges_test)
        else:
            return (df_nodes_train, df_nodes_test)
    
    
    def __call__(self, raw_data_file):
        print('\nPreprocessing data...', end='')
        raw_df = self.load_data(raw_data_file)
        if self.bank is not None:
            bank_df = raw_df[raw_df['bankOrig'] == self.bank]
            preprocessed_df = self.preprocess(bank_df, self.bank)
            print(' done\n')
            return preprocessed_df
        else:
            preprocessed_dfs = []
            banks = raw_df['bankOrig'].unique()
            for bank in banks:
                bank_df = raw_df[raw_df['bankOrig'] == bank]
                preprocessed_df = self.preprocess(bank_df, bank)
                preprocessed_dfs.append(preprocessed_df)
            print(' done\n')
            return preprocessed_dfs
        