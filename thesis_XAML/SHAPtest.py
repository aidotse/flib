import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import shap

n_samples = 2*50

data_0 = np.random.rand(50,2)
#data_0[:,0] = -data_0[:,0]
data_1 = -np.random.rand(50,2)

plt.figure()
plt.scatter(data_0[:,0], data_0[:,1], label='Class 0', marker='o')
plt.scatter(data_1[:,0], data_1[:,1], label='Class 1', marker='x')
plt.savefig('plot.png')

class LogisticRegressor(torch.nn.Module):
    def __init__(self, input_dim=23, output_dim=2):
        super(LogisticRegressor, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim-1)
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        outputs = x #torch.cat((1.0 - x, x))
        return outputs
    
    def predict(self, x):
        x = self.forward(x)
        return torch.round(x)
        

# Assuming data and labels are torch tensors
data_all = np.concatenate((data_0, data_1))
data = torch.from_numpy(data_all)
labels_all = np.concatenate((np.zeros((50,1)), np.ones((50,1))))
labels = torch.from_numpy(labels_all)

# Hyperparameters
lr = 0.1
num_epochs = 1000

# Instantiate the model, optimizer, and loss function
model = LogisticRegressor(2,2)

data = data.float()
labels = labels.float()

# print(data[0])
# print(model.forward(data[0]))

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    # Forward pass
    outputs = model(data)
    loss = loss_fn(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

print('Done!')

predictions = np.zeros((n_samples,1))
for i in range(n_samples):
    with torch.no_grad():
        predictions[i] = model.predict(data[i]).detach().numpy().astype(int)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels = labels.numpy().astype(int)
print(type(labels))
print(type(predictions))
print(np.shape(labels))
print(np.shape(predictions))

print(labels)
print(predictions)

correct_predictions = 0
incorrect_predictions = 0
for i in range(n_samples):
    if predictions[i] == labels[i]:
        correct_predictions = correct_predictions + 1
    else:
        incorrect_predictions = incorrect_predictions + 1

accuracy = correct_predictions/n_samples

print('Accuracy = {}'.format(accuracy))
#Note: First of all the model overfitts and then we calculate the
#'accuracy' using the _training_ data. This is only for testing SHAP
#values really and seeing that they make sense.

# Calculate SHAP-values
f = lambda x: model( Variable( torch.from_numpy(x) ) ).detach().numpy()
data = data.numpy()
explainer = shap.KernelExplainer(f, data)
shap_values = explainer.shap_values(data)
feature_names = ['x','y']
plt.clf()
shap.summary_plot(shap_values[0], data, feature_names, show=True)
plt.savefig('shap_values_toyexample.png')


print(type(shap_values))
print(len(shap_values))
for sv in shap_values:
    print(type(sv))
    print(sv.shape)

plt.clf()
expected_value = explainer.expected_value
shap.decision_plot(expected_value, shap_values[0], feature_names)
plt.savefig('decision_plot_toyexample.png')

plt.clf()
shap.force_plot(
    expected_value,
    shap_values[0][33,:],
    feature_names,
    link="logit",
    matplotlib=True,
)
plt.savefig('force_plot_toyexample.png')

plt.clf()
explainer = shap.KernelExplainer(f, data)
sv = explainer(data)
exp = shap.Explanation(sv[:,:,0], sv.base_values[:,0], data, feature_names=None)
shap.waterfall_plot(exp[0])
plt.savefig('waterfall_plot_toyexample.png')

