import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GINEConv, GATConv, BatchNorm, Linear
from torch_geometric.data import Data
#from dgl.nn.pytorch.conv import GATConv
import numpy as np 
from collections import deque
import copy


# (Written by Tomas & Agnes)
class LogisticRegressor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
    def class_probabilities(self, x):
        with torch.no_grad():
            x = self.forward(x)
            class_probs = torch.cat((1-x, x), dim = 1)
        return class_probs.reshape(-1,2)
    
    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            predicted_class = x.detach().round()
        return predicted_class.reshape(-1,1)


# Insert decision tree model here


# Insert random forest model here


# Insert MLP model here


# (Written by Edvin, edited by Tomas & Agnes)
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        convs = [GCNConv(input_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + [GCNConv(hidden_dim, output_dim)]
        self.convs = torch.nn.ModuleList(convs)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)])
        self.dropout = dropout
        self.softmax = torch.nn.Softmax(dim=1)
        self.testdata = []
        self.node_to_explain = []

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
            
        #Note: When using CrossEntropyLoss, the softmax function is included in the loss function
        out = x
            
        return out
    
        # Adaption for Node Feature Vector (NFV) input
    def set_test_data(self, testdata):
        self.testdata = testdata
    
    def set_node_to_explain(self, node_to_explain):
        self.node_to_explain = node_to_explain
    
    def forward_NFVinput(self, node_feature_vec):
        print('Starting forward_LIME...')
        num_nodes = self.testdata.x.shape[0]
        
        node_feature_vec = node_feature_vec.reshape(-1,self.input_dim)
        num_samples = node_feature_vec.shape[0]
        out = torch.zeros((num_samples,2))
        print(f'Number of samples = {num_samples}')
        
        data_list = []
        
        print('Loading data...')
        for i in range(num_samples):
            new_graph = copy.deepcopy(self.testdata)
            new_graph.x[self.node_to_explain,:] = node_feature_vec[i,:]
            data_list.append(new_graph)
        print(f'number of graphs = {len(data_list)}')
        
        print('Loading data into a single batch...')
        #dataset = CustomDataset(data_list)
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        
        print('Starting forward pass...')
        with torch.no_grad():
            out_tmp = self.softmax(self.forward(batch))
        
        print(f'out_tmp.shape = {out_tmp.shape}')
        print('Extracting output...')
        for i in range(batch.num_graphs):
            out[i] = out_tmp[self.node_to_explain+i*num_nodes,:]
            
        print('Finished.')

        return out


# (Written by Edvin)
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
        return x#torch.softmax(x, dim=-1)


# (Written by Tomas & Agnes)
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat = False, dropout=dropout)

    def forward(self, x, edge_index):
        x, attention_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attention_weights2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attention_weights3 = self.conv3(x, edge_index, return_attention_weights=True)
        
        #Note: When using CrossEntropyLoss, the softmax function is included in the loss function
        #out = self.softmax(x)
        return x, attention_weights1, attention_weights2, attention_weights3


# class GCN_NFVinput(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
#         super(GCN_NFVinput, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         convs = [GCNConv(input_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + [GCNConv(hidden_dim, output_dim)]
#         self.convs = torch.nn.ModuleList(convs)
#         self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)])
#         self.dropout = dropout
#         self.softmax = torch.nn.Softmax(dim=1)
#         self.testdata = []
#         self.node_to_explain = []

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         for i, layer in enumerate(self.convs):
#           x = layer(x, edge_index)
#           if i < len(self.convs)-1:
#             x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
            
#         #Note: When using CrossEntropyLoss, the softmax function is included in the loss function
#         #out = self.softmax(x)
#         out = x
#         return out
    
#     # Adaption for Node Feature Vector (NFV) input
#     def set_test_data(self, testdata):
#         self.testdata = testdata
    
#     def set_node_to_explain(self, node_to_explain):
#         self.node_to_explain = node_to_explain
    
#     def forward_NFVinput(self, node_feature_vec):
#         print('Starting forward_LIME...')
#         num_nodes = self.testdata.x.shape[0]
        
#         node_feature_vec = node_feature_vec.reshape(-1,self.input_dim)
#         num_samples = node_feature_vec.shape[0]
#         out = torch.zeros((num_samples,2))
        
#         data_list = []
        
#         print('Loading data...')
#         for i in range(num_samples):
#             new_graph = copy.deepcopy(self.testdata)
#             new_graph.x[self.node_to_explain,:] = node_feature_vec[i,:]
#             data_list.append(new_graph)
#         print(f'number of graphs = {len(data_list)}')
        
#         print('Loading data into a single batch...')
#         #dataset = CustomDataset(data_list)
#         batch = torch_geometric.data.Batch.from_data_list(data_list)
        
#         print('Starting forward pass...')
#         with torch.no_grad():
#             out_tmp = self.softmax(self.forward(batch))
        
#         print(f'out_tmp.shape = {out_tmp.shape}')
#         print('Extracting output...')
#         for i in range(batch.num_graphs):
#             out[i] = out_tmp[self.node_to_explain+i*num_nodes,:]
            
#         print('Finished.')

#         return out
        
    # def forward_LIME_OLD(self, node_feature_vec):
    #     print('Starting forward_LIME...')
    #     print(f'node_feature_vec type = {type(node_feature_vec)}')
    #     print(f'node_feature_vec is on device: {node_feature_vec.device}')
    #     print(f'self.testdata is on device: {self.testdata.x.device}')
    #     print(node_feature_vec.shape)
    #     node_feature_vec = node_feature_vec.reshape(-1,self.input_dim)
    #     print(node_feature_vec.shape)
    #     out = torch.zeros((node_feature_vec.shape[0],2))
    #     for i in range(node_feature_vec.shape[0]):
    #         self.testdata.x[self.node_to_explain] = node_feature_vec[i,:]
    #         with torch.no_grad():
    #             out_tmp = self.softmax(self.forward(self.testdata))
    #             out[i] = out_tmp[self.node_to_explain]
    #         if i % 100 == 0:
    #             print('LIME progress: ', i, '/', node_feature_vec.shape[0])
    #     return out




# (Written by Edvin, edited by Tomas & Agnes)
class GCN_GNNExplainer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_GNNExplainer, self).__init__()
        self.output_dim = output_dim
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

    def forward(self, x, edge_index):
        x, edge_index = x, edge_index
        for i, layer in enumerate(self.convs):
          x = layer(x, edge_index)
          if i < len(self.convs)-1:
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        #Note: When using CrossEntropyLoss, the softmax function is included in the loss function
        #out = self.softmax(x)
        out = x
            
        return out


# (Written by Edvin, edited by Tomas & Agnes)
class GCN_GraphSVX(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_GraphSVX, self).__init__()
        self.output_dim = output_dim
        convs = [GCNConv(input_dim, hidden_dim)] + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)] + [GCNConv(hidden_dim, output_dim)]
        self.convs = torch.nn.ModuleList(convs)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)])
        self.dropout = dropout
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        x, edge_index = x, edge_index
        for i, layer in enumerate(self.convs):
          x = layer(x, edge_index)
          if i < len(self.convs)-1:
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        #Note: When using LogSoftmax function in the final layer we need
        #to use NLLLoss instead of cross entropy loss
        out = self.log_softmax(x)
        #out = x
            
        return out




# (Written by Tomas & Agnes)
class GAT_GraphSVX(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout=0.3):
        super(GAT_GraphSVX, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat = False, dropout=dropout)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.return_attention_weights = False
    
    def set_return_attention_weights(self, return_attention_weights):
        if return_attention_weights == True or return_attention_weights == False:
            self.return_attention_weights = return_attention_weights
        else:
            raise ValueError('return_attention_weights must be either True or False')


    def forward(self, x, edge_index):
        x, attention_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attention_weights2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attention_weights3 = self.conv3(x, edge_index, return_attention_weights=True)
        x = self.log_softmax(x) #<--- need to use NLLLoss instead of CrossEntropyLoss

        if self.return_attention_weights:
            return x, attention_weights1, attention_weights2, attention_weights3
        else:
            return x
    
    # Adaption for Node Feature Vector (NFV) input
    def set_test_data(self, testdata):
        self.testdata = testdata
    
    def set_node_to_explain(self, node_to_explain):
        self.node_to_explain = node_to_explain
    
    def forward_NFVinput(self, node_feature_vec):
        print('Starting forward_LIME...')
        num_nodes = self.testdata.x.shape[0]
        
        node_feature_vec = node_feature_vec.reshape(-1,self.in_channels)
        num_samples = node_feature_vec.shape[0]
        out = torch.zeros((num_samples,2))
        print(f'Number of samples = {num_samples}')
        
        data_list = []
        
        print('Loading data...')
        for i in range(num_samples):
            new_graph = copy.deepcopy(self.testdata)
            new_graph.x[self.node_to_explain,:] = node_feature_vec[i,:]
            data_list.append(new_graph)
        print(f'number of graphs = {len(data_list)}')
        
        print('Loading data into a single batch...')
        #dataset = CustomDataset(data_list)
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        
        print('Starting forward pass...')
        with torch.no_grad():
            out_tmp = self.forward(batch.x, batch.edge_index).exp()
        
        print(f'out_tmp.shape = {out_tmp.shape}')
        print('Extracting output...')
        for i in range(batch.num_graphs):
            out[i] = out_tmp[self.node_to_explain+i*num_nodes,:]
            
        print('Finished.')

        return out


class GraphSAGE_GraphSVX_foroptuna(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels ,dropout=0.2, return_type='logits'):
        super().__init__()
        self.dropout = dropout
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.node_emb = Linear(in_channels, hidden_channels)
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.classifier = Linear(hidden_channels, out_channels)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.return_type = return_type

    def forward(self, x, edge_index):
        x = self.node_emb(x)
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        
        x = self.classifier(x)
        if self.return_type=="log_probas":
            x=self.log_softmax(x)
            
        return x#torch.softmax(x, dim=-1)

    def set_return_type(self, return_type):
        if return_type == 'logits' or return_type == 'log_probas':
            self.return_type = return_type
        else:
            raise ValueError('return_type must be either "logits" or "log_probas"')
        


# (Written by Tomas & Agnes)
class GAT_GraphSVX_foroptuna(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout=0.3):
        super(GAT_GraphSVX_foroptuna, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat = False, dropout=dropout)
        self.MLP_post1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.MLP_post2 = torch.nn.Linear(hidden_channels, out_channels)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.return_attention_weights = False
        self.return_type = 'logits'
        self.testdata = []
        self.node_to_explain = []
        
    
    def set_return_attention_weights(self, return_attention_weights):
        if return_attention_weights == True or return_attention_weights == False:
            self.return_attention_weights = return_attention_weights
        else:
            raise ValueError('return_attention_weights must be either True or False')
    
    def set_return_type(self, return_type):
        if return_type == 'logits' or return_type == 'log_probas':
            self.return_type = return_type
        else:
            raise ValueError('return_type must be either "logits" or "log_probas"')

    def forward(self, x, edge_index):
        x, attention_weights1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attention_weights2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attention_weights3 = self.conv3(x, edge_index, return_attention_weights=True)
        x = self.MLP_post1(x)
        x = self.MLP_post2(x)
        if self.return_type == "log_probas":
            x = self.log_softmax(x) #<--- log probas should only be used when running the explainers.
        
        if self.return_attention_weights:
            return x, attention_weights1, attention_weights2, attention_weights3
        else:
            return x
    
    def forward_NFVinput(self, node_feature_vec):
        print('Starting forward_NFVinput...')
        
        print('Setting return type of forward pass to "logits" so that forward_NFVinput returns "probas"...')
        self.return_type = 'logits'
        
        num_nodes = self.testdata.x.shape[0]
        
        node_feature_vec = node_feature_vec.reshape(-1,self.in_channels)
        num_samples = node_feature_vec.shape[0]
        out = torch.zeros((num_samples,2))
        print(f'Number of samples = {num_samples}')
        
        data_list = []
        
        print('Loading data...')
        for i in range(num_samples):
            new_graph = copy.deepcopy(self.testdata)
            new_graph.x[self.node_to_explain,:] = node_feature_vec[i,:]
            data_list.append(new_graph)
        print(f'number of graphs = {len(data_list)}')
        
        print('Loading data into a single batch...')
        #dataset = CustomDataset(data_list)
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        
        print('Starting forward pass...')
        with torch.no_grad():
            out_tmp = F.softmax(self.forward(batch.x, batch.edge_index), dim = 1)
        
        print(f'out_tmp.shape = {out_tmp.shape}')
        print('Extracting output...')
        for i in range(batch.num_graphs):
            out[i] = out_tmp[self.node_to_explain+i*num_nodes,:]
            
        print('Finished.')

        return out
    
    # Adaption for Node Feature Vector (NFV) input
    def set_test_data(self, testdata):
        self.testdata = testdata
    
    def set_node_to_explain(self, node_to_explain):
        self.node_to_explain = node_to_explain
    
    def forward_NFVinput(self, node_feature_vec):
        print('Starting forward_NFVinput...')
        
        print('Setting return type of forward pass to "logits" so that forward_NFVinput returns "probas"...')
        self.return_type = 'logits'
        
        num_nodes = self.testdata.x.shape[0]
        
        node_feature_vec = node_feature_vec.reshape(-1,self.in_channels)
        num_samples = node_feature_vec.shape[0]
        out = torch.zeros((num_samples,2))
        print(f'Number of samples = {num_samples}')
        
        data_list = []
        
        print('Loading data...')
        for i in range(num_samples):
            new_graph = copy.deepcopy(self.testdata)
            new_graph.x[self.node_to_explain,:] = node_feature_vec[i,:]
            data_list.append(new_graph)
        print(f'number of graphs = {len(data_list)}')
        
        print('Loading data into a single batch...')
        #dataset = CustomDataset(data_list)
        batch = torch_geometric.data.Batch.from_data_list(data_list)
        
        print('Starting forward pass...')
        with torch.no_grad():
            out_tmp = F.softmax(self.forward(batch.x, batch.edge_index), dim = 1)
        
        print(f'out_tmp.shape = {out_tmp.shape}')
        print('Extracting output...')
        for i in range(batch.num_graphs):
            out[i] = out_tmp[self.node_to_explain+i*num_nodes,:]
            
        print('Finished.')

        return out