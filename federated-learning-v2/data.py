import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class BankDataset():
    def __init__(self, path=None):
        self.trainsets = [] 
        self.valsets = [] 
        self.testsets = []
        if path:
            self.load_data(path)

    def load_data(self, path, create_valsets=False, val_size=0.2, test_size=0.2, seed=42):
        files = os.listdir(path)
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(path, file)
                df = pd.read_csv(file_path)
                df = df.drop(columns='account').sample(frac=1, random_state=seed).reset_index(drop=True)
                if create_valsets:
                    trainset, valtestset = train_test_split(df, test_size=val_size+test_size, random_state=seed)
                    valset, testset = train_test_split(valtestset, test_size=test_size/(val_size+test_size), random_state=seed)
                    self.valsets.append(valset)
                else:
                    trainset, testset = train_test_split(df, test_size=test_size, random_state=seed)
                self.trainsets.append(trainset)
                self.testsets.append(testset)
        self.testsets = [pd.concat(self.testsets).reset_index(drop=True)]

    def datasets(self):
        return self.trainsets, self.valsets, self.testsets


