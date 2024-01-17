import torch
from torch.nn import functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GINEConv, BatchNorm, Linear
from torch_geometric.data import Data

class GCNLPA(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_layers, dropout, edge_dim, k, device):
        super(GCNLPA, self).__init__()
        self.device = device
        convs = [GCNConv(input_dim, hidden_dim)]
        convs += [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)]
        convs += [GCNConv(hidden_dim, output_dim)]
        self.convs = torch.nn.ModuleList(convs)
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)]
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = dropout
        self.edge_weight = torch.nn.Parameter(torch.ones(edge_dim))
        self.k = k
        
    
    def forward(self, data, adj_t=None):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.convs):
          x = layer(x, edge_index, self.edge_weight.sigmoid())
          if i < len(self.convs)-1:
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.softmax(x)
        # LPA implementation with dense format
        labels = torch.nn.functional.one_hot(data.y.type(torch.long)).type(torch.float)
        matrix = torch_geometric.utils.to_dense_adj(
            data.edge_index, 
            edge_attr=self.edge_weight.sigmoid(), 
            max_num_nodes=data.num_nodes
        )
        matrix = matrix.squeeze(0)
        selfloop = torch.diag(torch.ones(matrix.shape[0])).to(self.device)
        matrix += selfloop
        for _ in range(self.k):
          y = torch.matmul(matrix, labels)
          labels = y
        return out, torch.nn.functional.normalize(labels, dim=1)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN, self).__init__()
        convs = [GCNConv(input_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + [GCNConv(hidden_dim, output_dim)]
        self.convs = torch.nn.ModuleList(convs)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)])
        self.dropout = dropout
        self.softmax = torch.nn.Softmax(dim=1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.convs):
          x = layer(x, edge_index)
          if i < len(self.convs)-1:
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.softmax(x)
        return out
    
class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.node_emb = Linear(input_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.classifier = Linear(hidden_dim, output_dim)
    
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

class LogisticRegressor(torch.nn.Module):
    def __init__(self, input_dim=23, output_dim=2):
        super(LogisticRegressor, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim-1)
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        outputs = torch.cat((1.0 - x, x), dim=1)
        return outputs

class GINe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, edge_updates=False, residual=True, edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        self.node_emb = torch.nn.Linear(num_features, n_hidden)
        self.edge_emb = torch.nn.Linear(edge_dim, n_hidden)

        self.convs = torch.nn.ModuleList()
        self.emlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = GINEConv(torch.nn.Sequential(
                torch.nn.Linear(self.n_hidden, self.n_hidden), 
                torch.nn.ReLU(), 
                torch.nn.Linear(self.n_hidden, self.n_hidden)
                ), edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(torch.nn.Sequential(
                torch.nn.Linear(3 * self.n_hidden, self.n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = torch.nn.Sequential(
            Linear(n_hidden*3, 50), 
            torch.nn.ReLU(), 
            torch.nn.Dropout(self.final_dropout),
            Linear(50, 25), 
            torch.nn.ReLU(), 
            torch.nn.Dropout(self.final_dropout),
            Linear(25, n_classes)
        )
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        src, dst = edge_index
        
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates: 
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = x
        
        return self.mlp(out)