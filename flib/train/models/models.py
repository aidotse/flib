import torch
import torch_geometric.nn
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
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return torch.softmax(x, dim=-1)


class GAT(torch.nn.Module):
    def __init__(self, input_dim:int, n_conv_layers:int, hidden_dim:int, output_dim:int, dropout:float=0.2):
        super().__init__()
        self.dropout = dropout
        self.input_layer = torch_geometric.nn.Linear(input_dim, hidden_dim)
        self.conv_layers = torch.nn.ModuleList([torch_geometric.nn.GATConv(hidden_dim, hidden_dim) for _ in range(n_conv_layers)])
        self.bns = torch.nn.ModuleList([torch_geometric.nn.BatchNorm(hidden_dim) for _ in range(n_conv_layers)])
        self.output_layer = torch_geometric.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x = self.input_layer(data.x)
        x = F.relu(x)
        for conv_layer, bn in zip(self.conv_layers, self.bns):
            x = conv_layer(x, data.edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.output_layer(x)
        return torch.softmax(x, dim=-1)
    
    def get_state_dict(self):
        return {key: value for key, value in self.state_dict().items() if 'bn' not in key}

    def set_state_dict(self, weights: dict) -> None:
        self.load_state_dict(weights, strict=False)
        

class GCN(torch.nn.Module):
    def __init__(self, input_dim:int, n_conv_layers:int, hidden_dim:int, output_dim:int, dropout:float=0.2, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.input_layer = torch_geometric.nn.Linear(input_dim, hidden_dim)
        self.conv_layers = torch.nn.ModuleList([torch_geometric.nn.GCNConv(hidden_dim, hidden_dim) for _ in range(n_conv_layers)])
        self.layer_norms = torch.nn.ModuleList([torch_geometric.nn.LayerNorm(hidden_dim) for _ in range(n_conv_layers)])
        self.output_layer = torch_geometric.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x = self.input_layer(data.x)
        x = F.relu(x)
        for conv_layer, layer_norm in zip(self.conv_layers, self.layer_norms):
            x = conv_layer(x, data.edge_index)
            x = layer_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.output_layer(x)
        return torch.softmax(x, dim=-1)
    
    def get_state_dict(self):
        return {key: value for key, value in self.state_dict().items() if 'layer_norms' not in key}

    def set_state_dict(self, weights: dict) -> None:
        self.load_state_dict(weights, strict=False)
    
    def gradients(self):
        return {name: param.grad.clone().detach() for name, param in self.named_parameters() if param.grad is not None and "layer_norm" not in name}
    
    def load_gradients(self, grads):
        for name, param in self.named_parameters():
            if param.grad is not None:
                param.grad = grads[name]


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim:int, n_conv_layers:int, hidden_dim:int, output_dim:int, dropout:float=0.2):
        super().__init__()
        self.dropout = dropout
        self.input_layer = torch_geometric.nn.Linear(input_dim, hidden_dim)
        self.conv_layers = torch.nn.ModuleList([torch_geometric.nn.SAGEConv(hidden_dim, hidden_dim) for _ in range(n_conv_layers)])
        self.bns = torch.nn.ModuleList([torch_geometric.nn.BatchNorm(hidden_dim) for _ in range(n_conv_layers)])
        self.output_layer = torch_geometric.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x = self.input_layer(data.x)
        x = F.relu(x)
        for conv_layer, bn in zip(self.conv_layers, self.bns):
            x = conv_layer(x, data.edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.output_layer(x)
        return torch.softmax(x, dim=-1)
    
    def get_state_dict(self):
        return {key: value for key, value in self.state_dict().items() if 'bn' not in key}

    def set_state_dict(self, weights: dict) -> None:
        self.load_state_dict(weights, strict=False)