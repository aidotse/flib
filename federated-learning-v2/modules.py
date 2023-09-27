import torch

class LogisticRegressor(torch.nn.Module):
    def __init__(self, input_dim=23, output_dim=1):
        super(LogisticRegressor, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        outputs = torch.cat((1.0 - x, x), dim=1)
        return outputs