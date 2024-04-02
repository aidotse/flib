import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Load data ---
df_train = pd.read_csv('../gnn/data/simulation2/swedbank/train/nodes.csv')
df_test = pd.read_csv('../gnn/data/simulation2/swedbank/test/nodes.csv')
# print(df_train.head())
# print(df_train['is_sar'])
# print(df_test.head())
# print(df_test['is_sar'])


# --- Define dataset class ---
class NodeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# --- Instantiate dataset ---
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

# Instanciate dataset
dataset_train = NodeDataset(features_train_tensor, labels_train_tensor)
dataset_test = NodeDataset(features_test_tensor, labels_test_tensor)
# Testing
# print(dataset_train.__len__())
# print(dataset_train.__getitem__(-1))
# print(dataset_test.__len__())
# print(dataset_test.__getitem__(-1))
# print(labels_train_tensor.size())


# --- Dataloader ---
batch_size = 32
dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size, shuffle=True)


# --- Create model class ---
class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        z = self.linear(x)
        a = F.sigmoid(z)
        a = torch.squeeze(a)

        return a


# --- Define function for training the network ---
n_samples = dataset_train.__len__()
n_batches = n_samples/batch_size

def train_epoch(model, optimizer, loss_fn, dataloader_train, device, print_every):
    print("Starting training...")
    model.train()
    train_loss_batches = []
    
    for batch_index, (x, y) in enumerate(dataloader_train, 1):
        
        features, labels = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model.forward(features)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        train_loss_batches.append(loss.item())
        
        if batch_index % print_every == 0:
            print("Batch {}/{}: Training loss = {}".format(batch_index,n_batches,train_loss_batches[-1]))

    return model, train_loss_batches


# --- Instantiate model ---
model_logistic_regression = LogisticRegression(10, 1)


# --- Choose loss function and optimizer, and load model to device ---
learning_rate = 1e-4

loss_function = nn.MSELoss()
optimizer = optim.Adam(model_logistic_regression.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))

model_logistic_regression.to(device)


# --- Train model ---
num_epochs = 100

for epoch in range(num_epochs):
    print('Epoch: {}/{}'.format(epoch,num_epochs))
    
    model_logistic_regression_trained = train_epoch(model = model_logistic_regression,
                                                    optimizer=optimizer,
                                                    loss_fn=loss_function,
                                                    dataloader_train=dataloader_train,
                                                    device=device,
                                                    print_every=10)
    print('Done!')

