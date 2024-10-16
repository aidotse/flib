import torch
import torch_geometric
from torch.nn import functional as F

class LogisticRegressor(torch.nn.Module):
    def __init__(self, input_dim=23, output_dim=2):
        super(LogisticRegressor, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim-1)
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        outputs = torch.cat((1.0 - x, x), dim=1)
        return outputs

class MLP(torch.nn.Module):
    def __init__(self, input_dim=23, n_hidden_layers=2, hidden_dim=64, output_dim=2):
        super(MLP, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return torch.softmax(x, dim=-1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.node_emb = torch_geometric.nn.Linear(input_dim, hidden_dim)
        self.conv1 = torch_geometric.nn.SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = torch_geometric.nn.SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = torch_geometric.nn.SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = torch_geometric.nn.BatchNorm(hidden_dim)
        self.bn2 = torch_geometric.nn.BatchNorm(hidden_dim)
        self.bn3 = torch_geometric.nn.BatchNorm(hidden_dim)
        self.classifier = torch_geometric.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x = self.node_emb(data.x)
        
        x = self.conv1(x, data.edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv2(x, data.edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv3(x, data.edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.classifier(x)
        return torch.softmax(x, dim=-1)