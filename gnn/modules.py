import torch
from torch.nn import functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
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

    def forward(self, data, adj_t=None):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.convs):
          x = layer(x, edge_index)
          if i < len(self.convs)-1:
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.softmax(x)
        return out