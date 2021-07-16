import torch

a = torch.tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.1770,  0.1098, -0.2033, -0.0770]])
b = torch.tensor([[False],
        [ True]])

print(a[1])