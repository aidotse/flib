import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric
from torch_geometric.typing import OptPairTensor, OptTensor
from torch import Tensor

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, BatchNorm, Linear
from torch_geometric.data import Data


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert targets to type long, as expected by nn.CrossEntropyLoss
        targets = targets.view(-1, 1).long()
        # Compute the log softmax over the inputs
        log_pt = F.log_softmax(inputs, dim=1)
        # Gather the log prob predictions for the correct classes
        log_pt = log_pt.gather(1, targets)
        log_pt = log_pt.view(-1)
        pt = log_pt.exp()
        
        # Compute the focal loss component
        focal_loss = -1 * (1 - pt) ** self.gamma * log_pt
        # Apply the alpha weighting
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = at * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Define three specific GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

        # Define batch normalization for each of the hidden layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def reset_parameters(self):
        # Reset parameters for all convolutional layers and batch normalization layers
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3 - Output layer
        x = self.conv3(x, edge_index)
        

        return x

class GCN2(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.dropout = dropout
        # Define just one GCNConv layer transitioning directly from input_dim to output_dim
        self.conv1 = GCNConv(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim) 

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Apply single GCN convolution layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

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
        return x

class GraphSAGE2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        # Initialize embeddings and convolutional layers
        self.node_emb = Linear(input_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)  # Only one convolution layer
        self.bn1 = BatchNorm(hidden_dim)  # Only one batch normalization layer
        self.classifier = Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x = self.node_emb(data.x)  # Embed input features
        
        # Single SAGE Convolution layer
        x = self.conv1(x, data.edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classifier to output features
        x = self.classifier(x)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        # First GAT convolution layer
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_channels * num_heads)  # Adjust for multi-head output

        # Second GAT convolution layer
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_channels * num_heads)  # Again, adjust for multi-head output

        # Third GAT convolution layer
        self.conv3 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.bn3 = BatchNorm(out_channels)  # Batch normalization for the output of the last GAT layer
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply the first GAT convolution layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)

        # Apply the second GAT convolution layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        # Apply the third GAT convolution layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)  # Applying ELU activation function after the last batch normalization
        
        return x
    
class GAT2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        # Only GAT convolution layer directly from in_channels to out_channels
        self.conv1 = GATConv(in_channels, out_channels, heads=num_heads, concat=False, dropout=dropout)
        self.bn1 = BatchNorm(out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply the GAT convolution layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        
        return x

class CustomGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0, bias=True):
        super().__init__(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout, bias=bias)
        self.class_weights = torch.nn.Parameter(torch.ones(2))  # Only two classes

    def forward(self, x, edge_index, size=None, return_attention_weights=None):
        # Perform the forward pass as usual
        x, attn = super().forward(x, edge_index, size=size, return_attention_weights=return_attention_weights)
        
        # Assuming x contains class logits or probabilities and the max index represents the class
        class_indices = torch.argmax(x, dim=-1)  # Get class index for each node
        class_weights = self.class_weights[class_indices]  # Get corresponding class weights
        
        # Apply the class-dependent weights to the attention scores
        attn = attn * class_weights.view(-1, 1).expand_as(attn)

        return x, attn if return_attention_weights else x
    
class GAT2NO_NORM(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        # Only GAT convolution layer directly from in_channels to out_channels
        self.conv1 = GATConv(in_channels, out_channels, heads=num_heads, concat=False, dropout=dropout)
        #self.bn1 = BatchNorm(out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply the GAT convolution layer
        x = self.conv1(x, edge_index)
        #x = self.bn1(x)
        x = F.relu(x)
        
        return x

class CustomGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        # Only GAT convolution layer directly from in_channels to out_channels
        self.conv1 = CustomGATConv(in_channels, out_channels, heads=num_heads, concat=False, dropout=dropout)
        #self.bn1 = BatchNorm(out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply the GAT convolution layer
        x = self.conv1(x, edge_index)
        #x = self.bn1(x)
        x = F.relu(x)
        
        return x

