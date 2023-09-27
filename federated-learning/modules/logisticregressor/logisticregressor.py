import torch
from torch.nn import Module, Linear
from torch import cat, relu, sigmoid
#import pandas as pd
#from data_transformer import DataTransformer

class LogisticRegressor(Module):
    def __init__(self, input_dim=23, output_dim=1):
        super(LogisticRegressor, self).__init__()
        self.linear = Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = sigmoid(x)
        outputs = cat((1.0 - x, x), dim=1)
        return outputs
'''
class LogisticRegressor():
    def __init__(self, train_data, discrete_columns=(), target_column=None):
        if target_column == None:
            self.target_column = train_data.columns[-1]
        else:
            self.target_column = target_column
        train_data = train_data.drop(columns=[target_column])
        self.data_transformer = DataTransformer()
        self.data_transformer.fit(train_data, discrete_columns)

    def fit(self, model, X, y, epochs=1):
        
        train_data = train_data.drop(columns=[self.target_column])
        train_data = self.data_transformer.transform(train_data)
        print(train_data)
        metrics = None
        return metrics, model.state_dict()
'''    
'''
df = pd.read_csv('data/adult/adult.csv')
df = df.drop(columns = ['fnlwgt', 'education'])
discrete_columns = ('workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income')
logreg = LogisticRegressor(df, discrete_columns)
model = Model()
logreg.fit(model, df)
'''