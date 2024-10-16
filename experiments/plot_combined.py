import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    
    data = '3_banks_homo_mid'
    
    metrics = ['precision_recall_curve', 'roc_curve']
    settings = ['centralized', 'isolated']
    clients = ['LogRegClient', 'DecisionTreeClient', 'RandomForestClient', 'GradientBoostingClient', 'SVMClient', 'KNNClient']
    dataset = 'test'
    
    for metric in metrics:
        for setting in settings:
            fig = plt.figure()
            for client in clients:
                csv = f'/home/edvin/Desktop/flib/experiments/results/{data}/{setting}/{client}/csv/{metric}_{dataset}.csv'
                df = pd.read_csv(csv)
                plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=f'{client}')
            plt.grid()
            plt.legend()
            if metric == 'precision_recall_curve':
                plt.title(f'Precision-recall curve ({setting})')
                plt.xlabel('recall')
                plt.ylabel('precision')
            elif metric == 'roc_curve':
                plt.plot([0, 1], [0, 1], linestyle='--', color='k')
                plt.title(f'ROC curve ({setting})')
                plt.xlabel('fpr')
                plt.ylabel('tpr')
            plt.savefig(f'plots/{metric}_{setting}.png')
            plt.close()
    
    data = '3_banks_homo_mid'
    metrics = ['accuracy', 'balanced_accuracy', 'recall', 'precision', 'f1', 'roc_curve', 'precision_recall_curve']
    clients = ['LogRegClient']
    settings = ['centralized', 'federated', 'isolated']
    client_ref = 'DecisionTreeClient'
    setting_ref = 'isolated'
    
    for metric in metrics:
        fig = plt.figure()
        for client in clients:
            for setting in settings:
                if metric == 'roc_curve' or metric == 'precision_recall_curve':
                    dataset = 'test'
                else:
                    dataset = 'val'
                csv = f'/home/edvin/Desktop/flib/experiments/results/{data}/{setting}/{client}/csv/{metric}_{dataset}.csv'
                df = pd.read_csv(csv)
                plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=f'{client} ({setting})')
        
        csv = f'/home/edvin/Desktop/flib/experiments/results/{data}/{setting_ref}/{client_ref}/csv/{metric}_{dataset}.csv'
        df = pd.read_csv(csv)
        # if df has one value
        if len(df) == 1:
            df = pd.DataFrame({'x': [0, 100], 'y': [df.iloc[0, 1], df.iloc[0, 1]]})
        plt.plot(df.iloc[:, 0], df.iloc[:, 1], 'k--', label=f'ref ({client_ref}, {setting_ref})')
        
        plt.legend()
        plt.grid()
        plt.savefig(f'plots/{metric}.png')
    

if __name__ == '__main__':
    main()