## Load dataset with and without SAR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import data
import scipy.stats as stats

class Datacheck():
    
    # init
    def __init__(self, dataset:str, importance):
        self.dataset = dataset
        self.load_data(dataset)
        if importance!='LLR':
            self.importance_train, self.importance_test = self.build_importance(self.dataset, usingTrain=True, importance=importance)
        # self.importance_train, self.importance_test = self.build_importance(self.dataset, usingTrain=True, importance=importance)
        self.normal_train
        self.normal_test
        self.withoutSAR_train
        self.withoutSAR_test

    def load_data(self,dataset):
        DATASET=dataset

        self.normal_train=pd.read_csv(f'data2/{DATASET}/bank/train/nodes.csv')
        self.normal_test=pd.read_csv(f'data2/{DATASET}/bank/test/nodes.csv')
        # self.normal_val=pd.read_csv(f'data/{DATASET}/bank/val/nodes.csv')
        self.withoutSAR_train=pd.read_csv(f'data2/{DATASET}_withoutSAR/bank/train/nodes.csv')
        self.withoutSAR_test=pd.read_csv(f'data2/{DATASET}_withoutSAR/bank/test/nodes.csv')
        # self.withoutSAR_val=pd.read_csv(f'data/{DATASET}_withoutSAR/bank/val/nodes.csv')

    
        
    def build_importance(self, dataset, usingTrain=True, importance='std'):
        #Importance can be diff_norm, std, or LLR
        
        n_train=pd.DataFrame(self.normal_train)
        n_test=pd.DataFrame(self.normal_test)
        w_train=pd.DataFrame(self.withoutSAR_train)
        w_test=pd.DataFrame(self.withoutSAR_test)

        cols=n_train.columns

        #Remove 'account' and 'bank'
        cols=cols[2:]

        diff_train = pd.merge(n_train,w_train, on='account', suffixes=('_1','_2'))
        diff_test = pd.merge(n_test,w_test,on='account', suffixes=('_1','_2'))
        # print('diff_test',diff_test.head())
        for col in cols:
            diff_train[col] = diff_train[col+'_1'] - diff_train[col+'_2']
            diff_test[col] = diff_test[col+'_1'] - diff_test[col+'_2']

        diff_train = diff_train.drop(diff_train.columns[diff_train.columns.str.endswith('_1')], axis=1)
        diff_train = diff_train.drop(diff_train.columns[diff_train.columns.str.endswith('_2')], axis=1)
        diff_test= diff_test.drop(diff_test.columns[diff_test.columns.str.endswith('_1')], axis=1)
        diff_test= diff_test.drop(diff_test.columns[diff_test.columns.str.endswith('_2')], axis=1)

        #save account order 
        accounts_train=diff_train['account']
        accounts_test=diff_test['account']

        # if (all(diff_train['account']!=diff_te['account'])):
        #     print('Error: accounts are not the same')
        #     return
        
        diff_train_final=diff_train.drop(['account','is_sar'], axis=1)
        diff_test_final=diff_test.drop(['account','is_sar'], axis=1)
        #Calculate importance using the mean and std

        if (importance=='std'):
        
            if (usingTrain):
                #calculate the std for the normal training set, not the difference
                # print(self.normal_train.head())
                std = self.normal_train.std(axis=0)
            else: 
                std = self.normal_test.std(axis=0)
                
            importance_train=diff_train_final/std
            importance_train['account']=accounts_train

            importance_test=diff_test_final/std
            importance_test['account']=accounts_test
        
        elif  (importance=='diff_norm'):

            if (usingTrain):
                diff_mean=diff_train_final.mean(axis=0)
                diff_std=diff_train_final.std(axis=0)
            else: 
                diff_mean=diff_test_final.mean(axis=0)
                diff_std=diff_test_final.std(axis=0)

            importance_train=(diff_train_final-diff_mean)/diff_std
            importance_train['account']=accounts_train

            importance_test=(diff_test_final-diff_mean)/diff_std
            importance_test['account']=accounts_test
            
        elif  (importance=='LLR'):
            #Calculate the log likelihood ratio using the 
            normal_accts=self.normal_train[self.normal_train['is_sar']==0]
            sar_accts=self.normal_train[self.normal_train['is_sar']==1]
            
            normal_mean=normal_accts[cols].mean(axis=0)
            normal_std=normal_accts[cols].std(axis=0)
            
            sar_mean=sar_accts[cols].mean(axis=0)
            sar_std=sar_accts[cols].std(axis=0)
            
            #Calculate the LLR for the self.normal_test and self.withoutSAR_test
            
            llr_values = []
            for idx, row in self.normal_test.iterrows():
                llr_row = []
                for col in cols:
                    x = row[col]
                    llr_normal = log_likelihood(x, normal_mean[col], normal_std[col])
                    llr_sar = log_likelihood(x, sar_mean[col], sar_std[col])
                    llr_row.append(llr_normal - llr_sar)
                llr_values.append(llr_row)
            llr_before=pd.DataFrame(llr_values, columns=cols)
            llr_values=[]
            for idx, row in self.withoutSAR_test.iterrows():
                llr_row = []
                for col in cols:
                    x = row[col]
                    llr_normal = log_likelihood(x, normal_mean[col], normal_std[col])
                    llr_sar = log_likelihood(x, sar_mean[col], sar_std[col])
                    llr_row.append(llr_normal - llr_sar)
                llr_values.append(llr_row)
            llr_after=pd.DataFrame(llr_values, columns=cols)
            
            # Replace 0 values in llr_after with NaN to avoid division by zero
            llr_after.replace(0, np.nan, inplace=True)

            # the importance is the llr_before divided by llr_after for each node. If division by 0, then the importance is 0
            
            importance_train=llr_before/llr_after
            importance_train['account']=accounts_train
            
            importance_test=llr_before/llr_after
            importance_test['account']=accounts_test
            
            
            # Replace NaN values resulting from division by zero with 0
            importance_train.fillna(0, inplace=True)
            importance_test.fillna(0, inplace=True)

            
            
        #Reorganize columns in dataframes
        cols = importance_train.columns.tolist()
        cols = cols[-1:]+cols[:-1]
        importance_train=importance_train[cols]
        importance_test=importance_test[cols]

        #Add is_SAR to the importance df
        importance_train['is_sar']=diff_train['is_sar']
        importance_test['is_sar']=diff_test['is_sar']

        return importance_train, importance_test


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

        #Sort by importance column
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
        
        #remove is sar
        llr_before=llr_before.drop(['is_sar'],axis=1)
        llr_after=llr_after.drop(['is_sar'],axis=1)
        
        importance=llr_before.T
        importance.columns=['LLR_before']
        importance['LLR_after']=llr_after.T
        importance['importance']=importance['LLR_before']/importance['LLR_after']
        
        importance['importance'].fillna(0, inplace=True)
        #put importance column first
        cols = importance.columns.tolist()
        cols = cols[-1:]+cols[:-1]
        importance=importance[cols]
        
        importance['importance']=-importance['importance']
        importance=importance.sort_values(by='importance',ascending=False)

        
        return importance
    
        

def log_likelihood(x, mean, std):
    
    # return stats.norm.logpdf(x, mean, std)
    return -0.5 * ((x - mean) / std)**2 - np.log(std) - 0.5 * np.log(2 * np.pi)
