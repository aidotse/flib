## Load dataset with and without SAR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import data



def load_data(dataset):
    DATASET=dataset

    normal_train=pd.read_csv(f'data/{DATASET}/bank/train/nodes.csv')
    normal_test=pd.read_csv(f'data/{DATASET}/bank/test/nodes.csv')

    withoutSAR_train=pd.read_csv(f'data/{DATASET}_withoutSAR/bank/train/nodes.csv')
    withoutSAR_test=pd.read_csv(f'data/{DATASET}_withoutSAR/bank/test/nodes.csv')

    return normal_train, normal_test, withoutSAR_train, withoutSAR_test

def build_importance(dataset, usingTrain=True):
    normal_train,normal_test,withoutSAR_train,withoutSAR_test=load_data(dataset)
    
    n_train=pd.DataFrame(normal_train)
    n_test=pd.DataFrame(normal_test)
    w_train=pd.DataFrame(withoutSAR_train)
    w_test=pd.DataFrame(withoutSAR_test)

    cols=n_train.columns

    #Remove 'account' and 'bank'
    cols=cols[2:]
  


    diff_train = pd.merge(n_train,w_train, on='account', suffixes=('_1','_2'))
    diff_test = pd.merge(n_test,w_test,on='account', suffixes=('_1','_2'))

    for col in cols:
        diff_train[col] = diff_train[col+'_1'] - diff_train[col+'_2']
        diff_test[col] = diff_test[col+'_1'] - diff_test[col+'_2']

    diff_train = diff_train.drop(diff_train.columns[diff_train.columns.str.endswith('_1')], axis=1)
    diff_train = diff_train.drop(diff_train.columns[diff_train.columns.str.endswith('_2')], axis=1)
    diff_test= diff_test.drop(diff_test.columns[diff_test.columns.str.endswith('_1')], axis=1)
    diff_test= diff_test.drop(diff_test.columns[diff_test.columns.str.endswith('_2')], axis=1)


    #save account order 
    accounts=diff_train['account']

    # if (all(diff_train['account']!=diff_te['account'])):
    #     print('Error: accounts are not the same')
    #     return
    
    diff_train_final=diff_train.drop(['account','is_sar'], axis=1)
    diff_test_final=diff_test.drop(['account','is_sar'], axis=1)

    #Calculate importance using the mean and std
    if (usingTrain):
        diff_mean=diff_train_final.mean(axis=0)
        diff_std=diff_train_final.std(axis=0)
    else: 
        diff_mean=diff_test_final.mean(axis=0)
        diff_std=diff_test_final.std(axis=0)

    # print('mean',diff_mean)
    # print('std',diff_std)
        
    importance_train=(diff_train_final-diff_mean)/diff_std
    importance_train['account']=accounts

    importance_test=(diff_test_final-diff_mean)/diff_std
    importance_test['account']=accounts

    #Reorganize columns in dataframes
    cols = importance_train.columns.tolist()
    cols = cols[-1:]+cols[:-1]
    importance_train=importance_train[cols]
    importance_test=importance_test[cols]

    #Add is_SAR to the importance df
    importance_train['is_sar']=diff_train['is_sar']
    importance_test['is_sar']=diff_test['is_sar']

    return importance_train, importance_test

def get_sar_accounts(dataframe):
    return dataframe[dataframe['is_sar']==1]['account'].index
    

def get_importance(node,dataframe):
    #Get node imoprtance using account_id
    account_id=dataframe.loc[node]['account']
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

    return feature_importance
    


def main():
    importance_train,importance_test=build_importance('10K_accts_MID5')

    print('hello')
    print(get_importance(node=0,dataframe=importance_train))
