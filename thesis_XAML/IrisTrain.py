import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
import pandas as pd
np.random.seed(1)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
np.random.seed(1)
import sys
import sklearn
import sklearn.ensemble
import anchor
from anchor import utils
from anchor import anchor_tabular
from modules import LogisticRegressor

def train_iris_model():

    
    iris = sklearn.datasets.load_iris()
    features = iris.data
    labels = iris.target

    features = iris.data
    labels = iris.target

    features = features[labels != 0] # Drop class 0
    features = features[:,2:] # Drop features 0 and 1

    labels = labels[labels != 0] # Drop class 0
    labels[labels == 2] = 0 #Switch name 


    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, train_size=0.80)

    train_np=train
    test_np=test

    feature_names = iris.feature_names[2:]
    class_names = iris.target_names[1:][::-1]


    train = torch.from_numpy(train)
    test = torch.from_numpy(test)

    labels_train = torch.from_numpy(labels_train).reshape(-1,1)
    labels_test = torch.from_numpy(labels_test).reshape(-1,1)

    model = LogisticRegressor(2,1)


    # Hyperparameters
    lr = 0.05
    num_epochs = 500

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    train = train.float()
    test = test.float()
    labels_train = labels_train.float()
    labels_test = labels_test.float()


    # Training loop
    for epoch in range(num_epochs):
        
        model.train()
        optimizer.zero_grad()
        outputs = model.forward(train)
        loss = loss_fn(outputs, labels_train)
        loss.backward()
        optimizer.step()
        
        # Calculate training and validation accuracy every epoch
        training_accuracy = 0
        test_accuracy = 0
        
        model.eval()
        outputs = model.predict(train)
        training_accuracy = (outputs == labels_train).float().mean()
        outputs = model.predict(test)
        test_accuracy = (outputs == labels_test).float().mean()
        #print('Training accuracy = {} | Test accuracy = {}'.format(training_accuracy, test_accuracy))
        
        # Print the loss every epoch
        #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}' + ' Training accuracy = {} | Test accuracy = {}'.format(training_accuracy, test_accuracy))
            
    #print('Done!')
    return model,train,test,feature_names,class_names