from main import *
import shap
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
import torch
import torch.nn.functional as F


#from __future__ import print_function

np.random.seed(1)

# Load data
df_train = pd.read_csv('/home/agnes/desktop/flib/data/simtest/swedbank/train/nodes.csv')
df_test = pd.read_csv('/home/agnes/desktop/flib/data/simtest/swedbank/test/nodes.csv')
 
# Extract features and labels
X_train = df_train.drop('is_sar', axis=1).values
y_train = df_train['is_sar']
X_test = df_test.drop('is_sar', axis=1).values
y_test = df_test['is_sar']

# Load or train a logistic regression model
if os.path.exists('./gnn/models/model_logreg.pt'):
    print('Using existing trained model.')
    model_logreg = torch.load('./gnn/models/model_logreg.pt')
    model_logreg.eval()
else:
    print('Training model from scratch.')
    model_logreg = train_logistic_regressor()
    torch.save(model_logreg,'./gnn/models/model_logreg.pt')
 
feature_names = df_train.drop('is_sar', axis = 1).columns
print(feature_names)
target_names = np.unique(y_train).astype(str).tolist()
print(target_names)
#target_names = ['is not sar', 'is sar']

# Move all data and model to GPU
device = torch.device('cuda:0')
print('Device: {}'.format(device))
model_logreg = model_logreg.to(device)

def predict_fn(x):
    model_logreg.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x).float().to(device)
        output = model_logreg(x_tensor)
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
    return probabilities.cpu().numpy()


explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=target_names, discretize_continuous=True)

i = np.random.randint(0, X_test.shape[0])
exp = explainer.explain_instance(X_test[i], predict_fn, num_features=10, top_labels=1)

available_labels = exp.available_labels()

# Use the first available label (modify this based on your use case)
label_to_use = available_labels[0]


# Get the explanation as a PyPlot figure
fig = exp.as_pyplot_figure(label=label_to_use)

fig.subplots_adjust(left=0.4, right=0.9)  # Adjust the left and right margins as needed


# Save the figure to a PNG file
fig.savefig('/home/agnes/desktop/flib/gnn/LIME-results/explanation.png')

