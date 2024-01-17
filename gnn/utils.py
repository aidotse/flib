import torch


def z_norm(data:torch.Tensor):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32), std)
    return (data - data.mean(0).unsqueeze(0)) / std