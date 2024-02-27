from main import *
import shap
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
 
 
# Load data
df_train = pd.read_csv('/home/agnes/desktop/flib/data/simtest/swedbank/train/nodes.csv')
df_test = pd.read_csv('/home/agnes/desktop/flib/data/simtest/swedbank/test/nodes.csv')
 
 
# Extract features and labels (converting to numpy arrays)
features_train = df_train.drop('is_sar', axis=1).values
labels_train = df_train['is_sar'].values
features_test = df_test.drop('is_sar', axis=1).values
labels_test = df_test['is_sar'].values
 
 
# Convert to pytorch tensors
features_train_tensor = torch.FloatTensor(features_train)
labels_train_tensor = torch.FloatTensor(labels_train)
features_test_tensor = torch.FloatTensor(features_test)
labels_test_tensor = torch.FloatTensor(labels_test)
 
 
# Load or train a logistic regression model
if os.path.exists('./gnn/models/model_logreg.pt'):
    print('Using existing trained model.')
    model_logreg = torch.load('./gnn/models/model_logreg.pt')
    model_logreg.eval()
else:
    print('Training model from scratch.')
    model_logreg = train_logistic_regressor()
    torch.save(model_logreg,'./gnn/models/model_logreg.pt')
 
 
# Move all data and model to GPU
device = torch.device('cuda:0')
print('Device: {}'.format(device))
features_train_tensor = features_train_tensor.to(device)
labels_train_tensor = labels_train_tensor.to(device)
features_test_tensor = features_test_tensor.to(device)
labels_test_tensor = labels_test_tensor.to(device)
model_logreg = model_logreg.to(device)
 
 
# --- Calculate SHAP values ---
# Get features
train_features_df = df_train.drop('is_sar', axis = 1) # pandas dataframe
test_features_df = df_test.drop('is_sar', axis = 1) # pandas dataframe
 
# Define function to wrap model to transform data to tensor
model_logreg.to(torch.device('cpu'))
f = lambda x: model_logreg( Variable( torch.from_numpy(x) ) ).detach().numpy()
 
# Convert my pandas dataframe to numpy
data = test_features_df.to_numpy(dtype=np.float32)
 
# The explainer doesn't like tensors, hence the f function
explainer = shap.KernelExplainer(f, data[0:100])
 
# Get the shap values from my test data
shap_values = explainer.shap_values(data[0:100])
 
# Enable the plots in jupyter
#shap.initjs()
 
feature_names = test_features_df.columns
# Plots
#shap.force_plot(explainer.expected_value, shap_values[0], feature_names)
#shap.dependence_plot("b1_price_avg", shap_values[0], data, feature_names)
#shap.summary_plot(shap_values[0], data[0:100], feature_names, show=True)
#plt.savefig('/home/agnes/desktop/flib/gnn/shap-results/summary_plot.png')


sv = explainer(train_features_df)
exp = Explanation(sv[:,:,1], sv.base_values[:,1], train_features_df, feature_names=None)
shap.plots.waterfall_ploot(exp[0])

plt.savefig('waaterfall.png')
#print(explainer.expected_value[0],shap_values[1])

#print(shap_values)
#shap.decision_plot(explainer.expected_value[1],shap_values[1],feature_names)
#plt.savefig('/home/agnes/desktop/flib/gnn/shap-results/decision2.png')

#shap.plots.bar(shap_values[0])
#plt.savefig('/home/agnes/desktop/flib/gnn/shap-results/bar_plot.png')

#one_sample=print(shap_values[0][0])

#shap.force_plot(explainer.expected_value[0], one_sample, train_features_df.iloc[0,:])
#plt.savefig('/home/agnes/desktop/flib/gnn/shap-results/force_plot.png')
