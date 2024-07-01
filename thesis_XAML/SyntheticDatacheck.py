## Load dataset with and without SAR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import data
import scipy.stats as stats
import torch


class Datacheck():
    
    # init
    def __init__(self, dataset:str, importance, aggregated = False,attention_weights=False,sar_indices_used=None):
        self.dataset = dataset
        self.load_data(dataset)
        self.attention_weights = attention_weights
        self.sar_indices_used=sar_indices_used

        self.importance_test = self.build_importance( importance=importance,aggregated=aggregated)
        # self.importance_train, self.importance_test = self.build_importance(self.dataset, usingTrain=True, importance=importance)
        self.normal_train
        self.normal_test
        # self.normal_val
        self.withoutSAR_train
        self.withoutSAR_test
        self.cols
        if importance=='llr':
            self.llr_before
            self.llr_after
        # self.withoutSAR_val
        self.diff_test_final
        # self.agg_normal_train=None
        # self.agg_normal_test
        # self.agg_withoutSAR

    def load_data(self,dataset):
        DATASET=dataset

        self.normal_train=pd.read_csv(f'/home/agnes/desktop/flib/thesis_XAML/ultra_final_data/{DATASET}/bank/train/nodes.csv')
        self.normal_test=pd.read_csv(f'/home/agnes/desktop/flib/thesis_XAML/ultra_final_data/{DATASET}/bank/test/nodes.csv')
        # self.normal_val=pd.read_csv(f'final_data/{DATASET}/bank/val/nodes.csv')
        self.withoutSAR_train=pd.read_csv(f'/home/agnes/desktop/flib/thesis_XAML/ultra_final_data/{DATASET}_withoutSAR/bank/train/nodes.csv')
        self.withoutSAR_test=pd.read_csv(f'/home/agnes/desktop/flib/thesis_XAML/ultra_final_data/{DATASET}_withoutSAR/bank/test/nodes.csv')
        # self.withoutSAR_val=pd.read_csv(f'final_data/{DATASET}_withoutSAR/bank/val/nodes.csv')

    
    
    def calculate_difference(self,normal,withoutSAR):

                
        diff=pd.merge(normal,withoutSAR,on='account',suffixes=('_1','_2'))
        diff.fillna(0,inplace=True) #If an account only exists in one of the dataframes, fill the missing values with 0
        
        # # Reorder diff_test based on the index of n_test
        # diff = diff.set_index('account').loc[normal.set_index('account').index].reset_index()

        for col in self.cols: 
            diff[col]=diff[col+'_1']-diff[col+'_2']
            
        diff=diff.drop(diff.columns[diff.columns.str.endswith('_1')], axis=1)
        diff=diff.drop(diff.columns[diff.columns.str.endswith('_2')], axis=1)
        
        return diff
    
    
    
    def aggregate_by_attention(self, data, train=False):
        
        edges = self.attention_weights[0]
        weights = self.attention_weights[1]

        edges = pd.DataFrame(edges.cpu().detach().numpy())
        weights = torch.mean(weights, dim=1).cpu().detach().numpy()

        # Initialize an empty DataFrame for importance_test
        importance_tmp = np.zeros((data.shape[0], len(self.cols)))

        # Convert DataFrame to numpy array for faster processing
        n_test_values = data[self.cols].values
        
        last_sar_index=self.sar_indices_used[-1]
        print('last_sar_index',last_sar_index)
        
        # Use 1000 random indices
        if train:
            self.random_indices = np.random.choice(data.shape[0], 1000, replace=False)
            #Order the random indices from small to large
            self.random_indices.sort()
            last_random_index=self.random_indices[-1]
            
        
            print('last_random_index',last_random_index)
            
            for idx in range(data.shape[0]):
                
                if idx in self.random_indices:
                    
                    conn_edges = edges.T[edges.T[1] == idx]
                    conn_nodes = conn_edges[0].values
                    edge_index = conn_edges.index.values
                    connected_weights = weights[edge_index]

                    # Iterate over columns and compute weighted sums separately
                    for col_idx, col in enumerate(self.cols):
                        weighted_sum = np.dot(n_test_values[conn_nodes, col_idx], connected_weights)
                        importance_tmp[idx, col_idx] = weighted_sum
                
                if idx>=last_random_index:
                    return pd.DataFrame(importance_tmp, columns=self.cols)
        else:
            for idx in range(data.shape[0]):
                
                if idx in self.sar_indices_used:
                    
                    conn_edges = edges.T[edges.T[1] == idx]
                    conn_nodes = conn_edges[0].values
                    edge_index = conn_edges.index.values
                    connected_weights = weights[edge_index]

                    # Iterate over columns and compute weighted sums separately
                    for col_idx, col in enumerate(self.cols):
                        weighted_sum = np.dot(n_test_values[conn_nodes, col_idx], connected_weights)
                        importance_tmp[idx, col_idx] = weighted_sum
                
                if idx>=last_sar_index:
                    return pd.DataFrame(importance_tmp, columns=self.cols)

        # return pd.DataFrame(importance_tmp, columns=self.cols)


        
    def build_importance(self, importance='std', aggregated=False):
        
        ''' Possible importances are std, llr, and att
        std= difference divided by std
        llr= log likelihood ratio before - after
        '''
        #Importance can be diff_norm, std, or LLR
        
        n_train=pd.DataFrame(self.normal_train)
        n_test=pd.DataFrame(self.normal_test)
        w_train=pd.DataFrame(self.withoutSAR_train)
        w_test=pd.DataFrame(self.withoutSAR_test)
        
        # print('before',n_test.head())
        # print(w_test.head())
        # #only keep the sar_indices_used in test
        # n_test=n_test.iloc[self.sar_indices_used]
        # w_test=w_test.iloc[self.sar_indices_used]
        # print('after',n_test.head())
        # print(w_test.head())
        
         
        cols=n_test.columns
        cols=cols[2:]
        self.cols=cols

        if aggregated: 
            n_account=n_test['account']
            w_account=w_test['account']
            
            if importance=='std':
                agg_normal_train=self.aggregate_by_attention(n_train,train=True)
                self.agg_normal_train=agg_normal_train

            
            agg_normal_test=self.aggregate_by_attention(n_test)
            agg_normal_test['account']=n_account

            agg_withoutSAR=self.aggregate_by_attention(w_test)
            agg_withoutSAR['account']=w_account
            
            self.agg_normal_test=agg_normal_test
            self.agg_withoutSAR=agg_withoutSAR
            diff_test=self.calculate_difference(agg_normal_test,agg_withoutSAR)
            
        else:

            diff_test=self.calculate_difference(n_test,w_test)
        accounts_test=diff_test['account']

        
        #save account order 
        
        self.diff_test_final=diff_test.drop(['account'], axis=1)
       

            
        # if aggregated:
        #     n_account=n_test['account']
        #     w_account=w_test['account']
            
        #     agg_normal_train=self.aggregate_by_attention(n_train)
        #     agg_normal_test=self.aggregate_by_attention(n_test)
        #     agg_normal_test['account']=n_account

        #     agg_withoutSAR=self.aggregate_by_attention(w_test)
        #     agg_withoutSAR['account']=w_account
            
        #     std=agg_normal_train.std(axis=0)
            
        #     print(std)
            
        #     diff_test=self.calculate_difference(agg_normal_test,agg_withoutSAR)
            
        #     self.diff_test_final=diff_test.drop(['account'], axis=1)
            
            
        #     importance_test=abs(self.diff_test_final/std)
            # importance_test['account']=accounts_test
         
            
        
        if (importance=='std'):
        
                #calculate the std for the normal training set, not the difference
                # print(self.normal_train.head())
            if aggregated: 
                std = agg_normal_train.std(axis=0)

            else:
                std = self.normal_train.std(axis=0)



            importance_test=abs(self.diff_test_final/std)
            importance_test['account']=accounts_test
            
            importance_test=importance_test[self.cols[0:-1]]
        
        
        # elif  (importance=='diff_norm'):

        #     if (usingTrain):
        #         diff_mean=diff_train_final.mean(axis=0)
        #         diff_std=diff_train_final.std(axis=0)
        #     else: 
        #         diff_mean=diff_test_final.mean(axis=0)
        #         diff_std=diff_test_final.std(axis=0)


        #     importance_test=(diff_test_final-diff_mean)/diff_std
        #     importance_test['account']=accounts_test
            
        elif (importance=='llr'):
            
            
            
            if aggregated: 
                print('agg',agg_normal_test.columns)
                agg_normal_test['is_sar']=n_test['is_sar']
                # agg_normal_train['is_sar']=n_train['is_sar']
                agg_withoutSAR['is_sar']=w_test['is_sar']
                
                normal_accts=agg_normal_test[agg_normal_test['is_sar']==0]
                sar_accts=agg_normal_test[agg_normal_test['is_sar']==1]
                node_features=agg_normal_test
                # n_train=pd.DataFrame(self.normal_train)
                
            else:
                        
                normal_accts=self.normal_train[self.normal_train['is_sar']==0]
                sar_accts=self.normal_train[self.normal_train['is_sar']==1]
                node_features=self.normal_test

                
            
            normal_mean=normal_accts[cols].mean(axis=0)
            normal_std=normal_accts[cols].std(axis=0)
            
            sar_mean=sar_accts[cols].mean(axis=0)
            sar_std=sar_accts[cols].std(axis=0)
            
            
            x=node_features
            
            llr_normal = log_likelihood(x, normal_mean, normal_std)
            llr_sar = log_likelihood(x, sar_mean, sar_std)
            
            #fill nan values with 0
            llr_normal.fillna(0, inplace=True)
            llr_sar.fillna(0, inplace=True)
            llr_before=llr_normal-llr_sar
            
            accounts_normal=self.normal_test['account'].values
            llr_before['account']=accounts_normal
            
            y=self.withoutSAR_test
            y=y[cols]
            llr_normal = log_likelihood(y, normal_mean, normal_std)
            llr_sar = log_likelihood(y, sar_mean, sar_std)
                
                 #fill nan values with 0
            llr_normal.fillna(0, inplace=True)
            llr_sar.fillna(0, inplace=True)
            llr_after=llr_normal-llr_sar
            accounts_withoutSAR=self.withoutSAR_test['account'].values
            llr_after['account']=accounts_withoutSAR
            
            self.llr_before=llr_before
            self.llr_after=llr_after
            importance_test=pd.merge(llr_before,llr_after, on='account', suffixes=('_1', '_2'), how='left')
            print('importance',importance_test)
            
            for col in cols:
                importance_test[col]=importance_test[col+'_2']-importance_test[col+'_1']
            #     importance_test[col] = importance_test.apply(lambda row: -row[col+'_1']/row[col+'_2'] if row[col+'_1'] <0 and row[col+'_2']>0 else row[col+'_1']/row[col+'_2'] , axis=1)
            #     importance_test[col] = importance_test.apply(lambda row: -row[col+'_1']/row[col+'_2'] if row[col+'_1'] >0 and row[col+'_2']>0 and row[col+'_1']>row[col+'_2'] else row[col+'_1']/row[col+'_2'] , axis=1)
            #     importance_test[col] = importance_test.apply(lambda row: -row[col+'_1']/row[col+'_2'] if row[col+'_1'] <0 and row[col+'_2']<0 and row[col+'_1']>row[col+'_2'] else row[col+'_1']/row[col+'_2'], axis=1)
            #     importance_test[col] = importance_test.apply(lambda row: 0 if row[col+'_1'] == row[col+'_2'] else row[col+'_1']/row[col+'_2'], axis=1)
                
                
                #here i want to make the importance[col] negative for some instances and positive for others. How to adjust
                

       
            importance_test=importance_test.drop(importance_test.columns[importance_test.columns.str.endswith('_1')], axis=1)
            importance_test=importance_test.drop(importance_test.columns[importance_test.columns.str.endswith('_2')], axis=1)
            
            importance_test=importance_test.drop(['account','is_sar'],axis=1)
        # importance_test=importance_test.drop(['std','sum','min','in_mean','out_min','in_median','sum_spending','out_sum','out_mean','mean_spending','median','max','std_spending','min_spending','out_median','in_std','count_in'],axis=1)

        #organize columns in order ['in_sum', 'mean', 'out_std', 'in_max', 'out_max', 'in_min', 'count_out','n_unique_in', 'n_unique_out', 'count_days_in_bank', 'count_phone_changes', 'median_spending', 'max_spending','count_spending']
            
        # importance_test=importance_test[['in_sum', 'mean', 'out_std', 'in_max', 'out_max', 'in_min', 'count_out','n_unique_in', 'n_unique_out', 'count_days_in_bank', 'count_phone_changes', 'median_spending', 'max_spending','count_spending']]

        # self.normal_test=self.normal_test[['in_sum', 'mean', 'out_std', 'in_max', 'out_max', 'in_min', 'count_out','n_unique_in', 'n_unique_out', 'count_days_in_bank', 'count_phone_changes', 'median_spending', 'max_spending','count_spending']]
        print('importance_test shape',importance_test.shape)
        print('importance cols',importance_test.columns)
            
        # #Reorganize columns in dataframes
        # cols = importance_test.columns.tolist()
        # # cols = cols[-1:]+cols[:-1]
        # importance_train=importance_train[cols]
        # importance_test=importance_test[cols]

        #Add is_SAR to the importance df
        # importance_train['is_sar']=diff_train['is_sar']
        # importance_test['is_sar']=diff_test['is_sar']

        return importance_test


    def get_sar_accounts(self):
        dataframe=self.normal_test
        return dataframe[dataframe['is_sar']==1]['account'].index.values
        

    def get_importance_several(self,nodes):
        #calculate importance for several nodes
        dataframe=self.importance_test
        
        #create an empty df with a column title for each node
        importance_df=pd.DataFrame(columns=nodes)

         

        for node in nodes: 
            self.get_importance(node)

  
    def get_node_importance(self,node):
        dataframe=self.importance_test
        #Get node imoprtance using account_id
        account_id=dataframe.loc[node]['account']
        print(f'Node: {node} is associated with Account_ID: {account_id} ')
        feature_importance=dataframe[dataframe['account']==account_id]
        feature_importance=feature_importance.abs()
        feature_importance=feature_importance.drop(['account','is_sar'],axis=1)
        
        #Transpose
        feature_importance=feature_importance.T

        #Set name for dataframe
        feature_importance.index.name=f'Node: {node}'

        #Change column name
        feature_importance.columns=['importance']

        #Sort by importance columnn
        feature_importance=feature_importance.sort_values(by='importance',ascending=False)

        #Add columns with org and without SAR values
        feature_importance['org']=self.normal_test[self.normal_test['account']==account_id].drop(['account','is_sar'],axis=1).T
        feature_importance['without_SAR']=self.withoutSAR_test[self.withoutSAR_test['account']==account_id].drop(['account','is_sar'],axis=1).T
        feature_importance['difference']=feature_importance['org']-feature_importance['without_SAR']
        return feature_importance
        
    def get_node_importance_LLR(self,node):
        
        normal_accts=self.normal_train[self.normal_train['is_sar']==0]
        sar_accts=self.normal_train[self.normal_train['is_sar']==1]
        n_train=pd.DataFrame(self.normal_train)
        
        cols=n_train.columns
        cols=cols[2:]
        
        account_id=self.normal_test.loc[node]['account']
        print(f'Node: {node} is associated with Account_ID: {account_id} in normal_test ')
        
        normal_mean=normal_accts[cols].mean(axis=0)
        normal_std=normal_accts[cols].std(axis=0)
        
        sar_mean=sar_accts[cols].mean(axis=0)
        sar_std=sar_accts[cols].std(axis=0)
               
        node_features=self.normal_test[self.normal_test['account']==account_id]
        x=node_features[cols]
        print(x)
        # print(x)
        llr_normal = log_likelihood(x, normal_mean, normal_std)
        llr_sar = log_likelihood(x, sar_mean, sar_std)
        #fill nan values with 0
        llr_normal.fillna(0, inplace=True)
        llr_sar.fillna(0, inplace=True)
        llr_before=llr_normal-llr_sar
        y=self.withoutSAR_test[self.withoutSAR_test['account']==account_id]
        y=y[cols]
        llr_normal = log_likelihood(y, normal_mean, normal_std)
        llr_sar = log_likelihood(y, sar_mean, sar_std)
        
        
         #fill nan values with 0
        llr_normal.fillna(0, inplace=True)
        llr_sar.fillna(0, inplace=True)
        llr_after=llr_normal-llr_sar
        
        
        #if  llr_after is empty 
        if llr_after.empty:
            llr_after=llr_before
        #remove is sar
        llr_before=llr_before.drop(['is_sar'],axis=1)
        llr_after=llr_after.drop(['is_sar'],axis=1)
        
        importance=llr_before.T
        importance.columns=['llr_before']
        importance['llr_after']=llr_after.T
        importance['importance']=importance['llr_before']/importance['llr_after']
        
        importance['importance'].fillna(0, inplace=True)
        #put importance column first
        cols = importance.columns.tolist()
        cols = cols[-1:]+cols[:-1]
        importance=importance[cols]
        
        
        importance['importance'] = importance.apply(lambda row: -row['importance'] if row['llr_before'] <0 and row['llr_after']>0 else row['importance'], axis=1)
        importance['importance'] = importance.apply(lambda row: -row['importance'] if row['llr_before'] >0 and row['llr_after']>0 and row['llr_before']>row['llr_after'] else row['importance'], axis=1)
        importance['importance'] = importance.apply(lambda row: -row['importance'] if row['llr_before'] <0 and row['llr_after']<0 and row['llr_before']>row['llr_after'] else row['importance'], axis=1)
        importance['importance'] = importance.apply(lambda row: 0 if row['llr_before'] == row['llr_after'] else row['importance'], axis=1)

        # print(importance)
        importance['org']=self.normal_test[self.normal_test['account']==account_id].drop(['account','is_sar'],axis=1).T
        
        #get without sar if existing in dataset
        if account_id in self.withoutSAR_test['account'].values:
            importance['without_SAR']=self.withoutSAR_test[self.withoutSAR_test['account']==account_id].drop(['account','is_sar'],axis=1).T
            importance['difference']=importance['org']-importance['without_SAR']
        else:
            importance['without_SAR']=0
            importance['difference']=0
        importance['difference']=importance['org']-importance['without_SAR']

        # importance['importance']=-importance['importance']
        importance=importance.sort_values(by='importance',ascending=False)

        
        return importance
    
        

def log_likelihood(x, mean, std):
    
    # return stats.norm.logpdf(x, mean, std)
    return -0.5 * ((x - mean) / std)**2 - np.log(std) - 0.5 * np.log(2 * np.pi)

#tensor([ 0.0902,  0.3820,  0.6490,  0.5217, -0.3474,  0.2930, -0.2239,  0.6200,
    #     0.1461,  0.6240, -0.0372, -0.6914,  1.0043, -0.3675], device='cuda:0')